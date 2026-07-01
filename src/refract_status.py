"""
refract_status — one-shot repo health summary, 100% deterministic, no LLM.

CLI:
    refract-status --root /path/to/repo
    refract-status --json | jq .tokens
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast_extractor as ae
from ast_extractor import count_tokens, extract_json
from refract_server import (
    _MAX_DEPTH,
    _detect,
    _function_nodes,
    _index_file,
    _iter_source_files,
    _read_source,
    _scan_function,
)

logger = logging.getLogger(__name__)

_LANG_DISPLAY = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "tsx": "TSX",
}


# ─────────────────────────────────────────────────────────────────────
# Core: single-pass collection
# ─────────────────────────────────────────────────────────────────────

def _ts_available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


def collect(root: str, max_depth: int = _MAX_DEPTH) -> dict:
    """Walk *root* once and return aggregated repo health data.

    Single pass — files are read and parsed exactly once.

    Returns:
      root          : str
      by_language   : {lang_key: {files, functions, classes, raw_tokens, compressed_tokens}}
      security      : {category: call_count}  — dangerous primitive counts by root module name
      ts_fallback   : [ext, …]               — extensions skipped due to missing tree-sitter
      errors        : {relpath: message}
    """
    root = os.path.normpath(os.path.abspath(root))
    if not os.path.isdir(root):
        return {"error": f"Not a directory: {root}"}

    by_language: dict[str, dict] = {}
    security: dict[str, int] = {}
    ts_fallback: set[str] = set()
    errors: dict[str, str] = {}

    ts_ok = _ts_available()

    for fpath in _iter_source_files(root, max_depth):
        relname = os.path.relpath(fpath, root)
        ext = os.path.splitext(fpath)[1].lower()
        backend, language = _detect(fpath)
        lang_key = language or "python"

        try:
            source = _read_source(fpath)
        except OSError as exc:
            errors[relname] = f"read error: {exc}"
            continue

        if backend == "ts" and not ts_ok:
            ts_fallback.add(ext)
            continue

        try:
            info = _index_file(source, backend, language)
        except Exception as exc:
            logger.warning("collect: skipping %s — %s", relname, exc)
            if backend == "ts":
                ts_fallback.add(ext)
            errors[relname] = f"{type(exc).__name__}: {exc}"
            continue

        entry = by_language.setdefault(lang_key, {
            "files": 0, "functions": 0, "classes": 0,
            "raw_tokens": 0, "compressed_tokens": 0,
        })
        entry["files"] += 1
        entry["functions"] += len(info["functions"])
        entry["classes"] += len(info["classes"])

        raw = count_tokens(source)
        entry["raw_tokens"] += raw
        try:
            if backend == "python":
                comp_text = ae.compress(source)
            else:
                import ts_extractor as ts
                comp_text = ts.compress(source, language or "javascript")
            entry["compressed_tokens"] += count_tokens(comp_text)
        except Exception:
            entry["compressed_tokens"] += raw  # 0% gain for this file

        if backend == "python":
            try:
                fonctions = extract_json(source)["fonctions"]
                tree = ast.parse(source)
                node_map = _function_nodes(tree)
                for fn in fonctions:
                    node = node_map.get(fn["nom"])
                    high_calls, med_calls = _scan_function(node, fn["appels"])
                    for call in high_calls + med_calls:
                        cat = call.split(".")[0]
                        security[cat] = security.get(cat, 0) + 1
            except Exception:
                pass

    return {
        "root": root,
        "by_language": by_language,
        "security": security,
        "ts_fallback": sorted(ts_fallback),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    return f"{n:,}"


def render_human(data: dict) -> str:
    if "error" in data:
        return f"Error: {data['error']}"

    lines: list[str] = []
    lines.append(f"Repo health — {data['root']}")
    lines.append("═" * 60)

    by_lang = data["by_language"]

    lines.append("\nFiles by language")
    if not by_lang:
        lines.append("  (no supported source files found)")
    else:
        lines.append(f"  {'Language':<14}{'Files':>6}  {'Functions':>9}  {'Classes':>7}")
        lines.append("  " + "─" * 40)
        tot_f = tot_fn = tot_cls = 0
        for lang_key in sorted(by_lang):
            s = by_lang[lang_key]
            label = _LANG_DISPLAY.get(lang_key, lang_key.capitalize())
            lines.append(
                f"  {label:<14}{s['files']:>6}  {s['functions']:>9}  {s['classes']:>7}"
            )
            tot_f += s["files"]
            tot_fn += s["functions"]
            tot_cls += s["classes"]
        lines.append("  " + "─" * 40)
        lines.append(f"  {'Total':<14}{tot_f:>6}  {tot_fn:>9}  {tot_cls:>7}")

    total_raw = sum(s["raw_tokens"] for s in by_lang.values())
    total_comp = sum(s["compressed_tokens"] for s in by_lang.values())
    gain = round((1 - total_comp / total_raw) * 100, 1) if total_raw else 0.0

    lines.append("\nTokens")
    lines.append(f"  Raw          {_fmt(total_raw):>10}")
    lines.append(f"  Compressed   {_fmt(total_comp):>10}")
    lines.append(f"  Gain                  {gain:.1f}%")

    lines.append("\nSecurity surface  (Python · AST · zero LLM)")
    sec = data["security"]
    if not sec:
        lines.append("  No dangerous calls detected")
    else:
        for cat, count in sorted(sec.items(), key=lambda x: -x[1]):
            plural = "s" if count != 1 else ""
            lines.append(f"  {cat:<22} {count:>3} call{plural}")

    lines.append("\nTree-sitter")
    fb = data["ts_fallback"]
    if not fb:
        lines.append("  All extensions supported")
    else:
        lines.append(f"  Fallback active for: {', '.join(fb)}")

    errs = data.get("errors", {})
    if errs:
        lines.append(f"\nParse errors  ({len(errs)} file(s) skipped)")
        for relpath, msg in list(errs.items())[:5]:
            lines.append(f"  {relpath}: {msg}")
        if len(errs) > 5:
            lines.append(f"  … and {len(errs) - 5} more")

    return "\n".join(lines)


def render_json(data: dict) -> str:
    by_lang = data.get("by_language", {})
    total_raw = sum(s["raw_tokens"] for s in by_lang.values())
    total_comp = sum(s["compressed_tokens"] for s in by_lang.values())
    gain = round((1 - total_comp / total_raw) * 100, 1) if total_raw else 0.0

    out: dict = {
        "root": data.get("root"),
        "by_language": by_lang,
        "tokens": {"raw": total_raw, "compressed": total_comp, "gain_pct": gain},
        "security": data.get("security", {}),
        "ts_fallback_extensions": data.get("ts_fallback", []),
    }
    if data.get("errors"):
        out["errors"] = data["errors"]
    if "error" in data:
        out["error"] = data["error"]
    return json.dumps(out, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refract-status",
        description=(
            "Refract — repo health summary.\n"
            "Reports files by language, token compression stats, security surface,\n"
            "and tree-sitter coverage. 100%% deterministic, zero LLM calls.\n\n"
            "Examples:\n"
            "  refract-status\n"
            "  refract-status --root /path/to/repo\n"
            "  refract-status --json | jq .tokens"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="Root directory to analyse (default: current directory).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output structured JSON instead of human-readable table.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: WARNING).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    root = os.path.normpath(os.path.abspath(args.root))
    data = collect(root)
    if args.as_json:
        print(render_json(data))
    else:
        print(render_human(data))
    if "error" in data:
        sys.exit(1)


if __name__ == "__main__":
    main()
