"""
refract_server — MCP server exposing Refract's compression directly as tools.

Unlike refract_proxy (which sits between an agent and another MCP server and
compresses *tool schemas*), this server exposes Refract's Python source
compression as three first-class MCP tools, so an agent can explore a codebase
cheaply:

  1. index_repo(path)              — walk a repo, return an aggregated index
                                     (functions, classes, imports, dependencies)
  2. get_compressed(file_path)     — S5-compress a single file + token stats
  3. expand(file_path, targets)    — return named defs verbatim + their
                                     compressed dependency context

Served over stdio (same pattern as RefractProxy.serve()), so it drops into
Claude Desktop / Cursor as a local MCP server.

CLI:
    refract-server --root /path/to/repo
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast_extractor
from ast_extractor import (
    IndexModule,
    compress,
    count_tokens,
    dependances,
    extract,
    extract_json,
)

logger = logging.getLogger(__name__)

# Directories never worth walking into.
_SKIP_DIRS = frozenset({"__pycache__", ".git", "venv", ".venv", "node_modules"})

# Walk no deeper than this many directory levels below the root.
_MAX_DEPTH = 3

# Extension → (backend, language). Python uses the stdlib ast extractor;
# everything else routes to the tree-sitter ts_extractor. (.tsx uses the tsx
# grammar so embedded JSX parses; it is still the TypeScript family.)
_EXT_BACKEND: dict[str, tuple[str, str | None]] = {
    ".py": ("python", None),
    ".js": ("ts", "javascript"),
    ".jsx": ("ts", "javascript"),
    ".mjs": ("ts", "javascript"),
    ".cjs": ("ts", "javascript"),
    ".ts": ("ts", "typescript"),
    ".tsx": ("ts", "tsx"),
}
_SUPPORTED_EXTS = frozenset(_EXT_BACKEND)


def _detect(file_path: str) -> tuple[str, str | None]:
    """Return (backend, language) for *file_path* based on its extension.

    Unknown extensions default to the Python backend (callers only ever pass
    paths already filtered to supported extensions, except direct file tools
    which then surface a normal parse error)."""
    ext = os.path.splitext(file_path)[1].lower()
    return _EXT_BACKEND.get(ext, ("python", None))


# ─────────────────────────────────────────────────────────────────────
# Path handling
# ─────────────────────────────────────────────────────────────────────
def _resolve(path: str, root: str) -> str:
    """Resolve *path* against *root* (absolute paths are kept as-is)."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(root, path))


def _read_source(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────
# Tool 1: index_repo
# ─────────────────────────────────────────────────────────────────────
def _iter_source_files(root: str, max_depth: int = _MAX_DEPTH):
    """Yield supported source files (.py/.js/.ts/.jsx/.tsx/…) under *root*,
    skipping noise dirs, capped at *max_depth*.

    Depth is measured in directory levels below *root* (root itself = 0).
    """
    root = os.path.normpath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= max_depth:
            dirnames[:] = []  # don't descend further, and skip files this deep
            continue
        # Prune noise dirs in place so os.walk skips them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in _SUPPORTED_EXTS:
                yield os.path.join(dirpath, name)


def _index_file(source: str, backend: str, language: str | None) -> dict:
    """Per-file index: function names, class names, imports (language-aware)."""
    if backend == "python":
        idx = IndexModule(ast.parse(source))
        return {
            "functions": sorted(idx.fonctions),
            "classes": sorted(idx.classes),
            "imports": sorted(idx.imports),
        }
    import ts_extractor as ts
    parsed = ts.extract(source, language or "javascript")
    return {
        "functions": sorted({f["nom"] for f in parsed["fonctions"]}),
        "classes": sorted(parsed["classes"]),
        "imports": sorted(set(parsed["imports"])),
    }


def index_repo(path: str, root: str = ".", max_depth: int = _MAX_DEPTH) -> dict:
    """Walk a repo and return an aggregated structural index.

    Runs the right extractor on every supported source file — Python (.py) via
    the stdlib AST, JavaScript/TypeScript (.js/.jsx/.ts/.tsx/.mjs/.cjs) via
    tree-sitter — at max depth 3, skipping ``__pycache__`` / ``.git`` /
    ``venv`` / ``node_modules``, and aggregates every function, class, import
    and dependency. A file that fails to parse is recorded in ``errors`` and
    skipped — it never aborts the walk.
    """
    base = _resolve(path, root)
    if not os.path.isdir(base):
        return {"error": f"Not a directory: {base}", "root": base}

    files: dict[str, dict] = {}
    errors: dict[str, str] = {}
    dependencies: set[str] = set()
    n_functions = n_classes = 0

    for fpath in _iter_source_files(base, max_depth):
        relname = os.path.relpath(fpath, base)
        backend, language = _detect(fpath)
        try:
            info = _index_file(_read_source(fpath), backend, language)
        except OSError as exc:
            errors[relname] = f"read error: {exc}"
            continue
        except Exception as exc:  # SyntaxError, tree-sitter missing, parse error
            logger.warning("index_repo: skipping %s — %s", relname, exc)
            errors[relname] = f"{type(exc).__name__}: {exc}"
            continue
        files[relname] = info
        dependencies.update(info["imports"])
        n_functions += len(info["functions"])
        n_classes += len(info["classes"])

    result = {
        "root": base,
        "files": files,
        "dependencies": sorted(dependencies),
        "totals": {
            "files": len(files),
            "functions": n_functions,
            "classes": n_classes,
            "dependencies": len(dependencies),
        },
    }
    if errors:
        result["errors"] = errors
    return result


# ─────────────────────────────────────────────────────────────────────
# Tool 2: get_compressed
# ─────────────────────────────────────────────────────────────────────
def get_compressed(file_path: str, root: str = ".") -> dict:
    """S5-compress a single source file and report token savings.

    Language is auto-detected from the extension: Python via the stdlib AST,
    JS/TS via tree-sitter. Parse / backend failures return an ``error`` dict
    rather than raising.
    """
    target = _resolve(file_path, root)
    backend, language = _detect(target)
    try:
        source = _read_source(target)
    except OSError as exc:
        return {"error": f"read error: {exc}", "file": target}

    try:
        if backend == "python":
            compressed = compress(source)
        else:
            import ts_extractor as ts
            compressed = ts.compress(source, language or "javascript")
    except SyntaxError as exc:
        return {"error": f"syntax error: {exc}", "file": target}
    except Exception as exc:  # tree-sitter missing / parse failure
        logger.warning("get_compressed: %s — %s", target, exc)
        return {"error": f"{type(exc).__name__}: {exc}", "file": target}

    tokens_before = count_tokens(source)
    tokens_after = count_tokens(compressed)
    reduction = (
        round((1 - tokens_after / tokens_before) * 100, 1) if tokens_before else 0.0
    )
    return {
        "file": target,
        "language": language or "python",
        "compressed": compressed,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "reduction_pct": reduction,
    }


# ─────────────────────────────────────────────────────────────────────
# Tool 3: expand
# ─────────────────────────────────────────────────────────────────────
def _node_source(node: ast.AST, source: str) -> str:
    """Verbatim source for a node, falling back to unparse if unavailable."""
    segment = ast.get_source_segment(source, node)
    return segment if segment is not None else ast.unparse(node)


def expand(file_path: str, targets: list[str], root: str = ".") -> dict:
    """Return named functions/classes verbatim + their dependency context.

    For each name in *targets* found at module level (or as a method inside a
    class), returns the full source plus the compressed dependency contract
    (data / type / internal / external) computed against the module vocabulary.
    Language is auto-detected from the extension (Python AST / JS-TS tree-sitter).
    """
    target_path = _resolve(file_path, root)
    backend, language = _detect(target_path)
    try:
        source = _read_source(target_path)
    except OSError as exc:
        return {"error": f"read error: {exc}", "file": target_path}

    if backend != "python":
        try:
            import ts_extractor as ts
            parsed = ts.expand(source, list(targets), language or "javascript")
        except Exception as exc:  # tree-sitter missing / parse failure
            logger.warning("expand: %s — %s", target_path, exc)
            return {"error": f"{type(exc).__name__}: {exc}", "file": target_path}
        return {
            "file": target_path,
            "targets": parsed["targets"],
            "missing": parsed["missing"],
            "context": {"imports": parsed["imports"]},
        }

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"error": f"syntax error: {exc}", "file": target_path}

    idx = IndexModule(tree)
    wanted = set(targets)
    found: dict[str, dict] = {}

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if node.name not in wanted or node.name in found:
            continue
        entry: dict = {
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "source": _node_source(node, source),
        }
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entry["dependencies"] = dependances(node, idx)
        found[node.name] = entry

    missing = sorted(wanted - set(found))
    # Module-level compressed contract for surrounding context.
    contract = extract(source)
    return {
        "file": target_path,
        "targets": found,
        "missing": missing,
        "context": {"imports": contract["imports"]},
    }


# ─────────────────────────────────────────────────────────────────────
# Tool 4: blast_radius
# ─────────────────────────────────────────────────────────────────────
def _risk_level(impacted_count: int) -> str:
    if impacted_count > 5:
        return "HIGH"
    if impacted_count > 2:
        return "MEDIUM"
    return "LOW"


def blast_radius(file_path: str, target_function: str, root: str = ".") -> dict:
    """Reverse-call-graph impact analysis for a Python function.

    Builds the call graph from ast_extractor.extract() (top-level functions and
    class methods alike), inverts the "appels" edges so each function points to
    the functions that call it, then BFS from *target_function* to find every
    function transitively affected by changing it.

    Pure graph traversal — zero LLM calls. Python files only.
    """
    target_path = _resolve(file_path, root)
    try:
        source = _read_source(target_path)
    except OSError as exc:
        return {"error": f"read error: {exc}", "file": target_path}

    try:
        graph = extract(source)
    except SyntaxError as exc:
        return {"error": f"syntax error: {exc}", "file": target_path}

    fonctions = graph["fonctions"]
    names = {f["nom"] for f in fonctions}

    if target_function not in names:
        return {
            "error": f"Function '{target_function}' not found in {target_path}",
            "available_functions": sorted(names),
            "file": target_path,
        }

    # Invert the call edges: callee → {callers}. A function F is "called by" C
    # whenever F appears in C's "appels" list.
    callers_of: dict[str, set[str]] = {}
    for fn in fonctions:
        caller = fn["nom"]
        for callee in fn["appels"]:
            callers_of.setdefault(callee, set()).add(caller)

    direct_callers = callers_of.get(target_function, set())

    # BFS upward through the reverse graph to collect the full blast radius.
    visited = {target_function}
    queue = deque([target_function])
    while queue:
        current = queue.popleft()
        for caller in callers_of.get(current, ()):  # noqa: SIM118 - set, not dict
            if caller not in visited:
                visited.add(caller)
                queue.append(caller)

    all_impacted = visited - {target_function}
    impacted_count = len(all_impacted)

    return {
        "target": target_function,
        "direct_callers": sorted(direct_callers),
        "all_impacted": sorted(all_impacted),
        "impacted_count": impacted_count,
        "total_functions": len(names),
        "risk_level": _risk_level(impacted_count),
        "file": target_path,
    }


# ─────────────────────────────────────────────────────────────────────
# Tool 5: security_surface
# ─────────────────────────────────────────────────────────────────────
# HIGH risk: code execution, deserialization, arbitrary import.
_HIGH_RISK = frozenset({
    "subprocess", "subprocess.run", "subprocess.Popen", "subprocess.call",
    "os.system", "os.popen", "os.execv", "os.execve",
    "eval", "exec", "compile",
    "pickle.loads", "pickle.load", "pickle.dumps",
    "__import__", "importlib.import_module",
    "ctypes",
})

# MEDIUM risk: network, file write, external data.
# NOTE: ``open`` is intentionally absent — it is handled separately because
# only write/append modes count (see _has_write_open).
_MEDIUM_RISK = frozenset({
    "socket", "socket.connect", "socket.bind",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.request",
    "httpx.get", "httpx.post", "httpx.Client",
    "urllib.request.urlopen", "urllib.urlopen",
    "paramiko", "ftplib", "smtplib",
})

# Mode-string characters that turn open() into a write/append operation.
_OPEN_WRITE_CHARS = ("w", "a", "x", "+")


def _matches(call: str, riskset: frozenset[str]) -> bool:
    """True if a dotted call name matches a risk set.

    Matches either exactly ("subprocess.run") or by root module ("subprocess"),
    so a bare-module entry like ``subprocess``/``ctypes``/``paramiko`` flags every
    attribute call on it (``subprocess.run``, ``ctypes.CDLL`` …), while a dotted
    entry like ``os.system`` only matches that exact call (not all of ``os.*``).
    """
    if call in riskset:
        return True
    return call.split(".")[0] in riskset


def _has_write_open(node: ast.AST) -> bool:
    """True if *node*'s body calls open() in a write/append mode.

    The mode is the 2nd positional arg or the ``mode=`` keyword. A literal string
    containing w/a/x/+ is a write. No mode → default ``"r"`` (read-only, skipped).
    A non-literal mode (a variable) can't be resolved statically, so it is flagged
    conservatively as a write.
    """
    for n in ast.walk(node):
        if not (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "open"
        ):
            continue
        mode = None
        if len(n.args) >= 2:
            mode = n.args[1]
        else:
            for kw in n.keywords:
                if kw.arg == "mode":
                    mode = kw.value
        if mode is None:
            continue  # default "r" → read-only, not a risk
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            if any(c in mode.value for c in _OPEN_WRITE_CHARS):
                return True
        else:
            return True  # dynamic mode → conservative flag
    return False


def _function_nodes(tree: ast.AST) -> dict:
    """Map function/method name → its AST node (first occurrence) for line hints."""
    nodes: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.setdefault(n.name, n)
    return nodes


def _scan_function(node, appels: list[str]) -> tuple[list[str], list[str]]:
    """Classify a function's calls into (high_risk_calls, medium_risk_calls)."""
    high: list[str] = []
    medium: list[str] = []
    for call in appels:
        if call == "open":
            continue  # mode-dependent, handled below
        if _matches(call, _HIGH_RISK):
            high.append(call)
        elif _matches(call, _MEDIUM_RISK):
            medium.append(call)
    if node is not None and "open" in appels and _has_write_open(node):
        medium.append("open")
    return high, medium


def security_surface(
    repo_path: str, root: str = ".", max_depth: int = _MAX_DEPTH
) -> dict:
    """Find functions that call dangerous primitives across a Python repo.

    Pure AST analysis — zero LLM calls. Walks Python files (max depth 3, same
    skips as index_repo), and for every function classifies its calls against a
    HIGH-risk set (code execution / deserialization / arbitrary import) and a
    MEDIUM-risk set (network / file-write / external data). ``open()`` is flagged
    medium only when called in a write/append mode (or with an undeterminable
    mode). A function with a HIGH-risk call is reported in ``high_risk`` only; one
    with only MEDIUM-risk calls in ``medium_risk``; a file with no risky function
    lands in ``clean``. Files that fail to parse are skipped and logged to stderr,
    never aborting the scan.
    """
    base = _resolve(repo_path, root)
    if not os.path.isdir(base):
        return {"error": f"Not a directory: {base}", "repo_path": base}

    high_risk: list[dict] = []
    medium_risk: list[dict] = []
    clean: list[str] = []
    total_functions = 0
    total_files = 0

    for fpath in _iter_source_files(base, max_depth):
        if not fpath.endswith(".py"):  # Python only for now
            continue
        relname = os.path.relpath(fpath, base)
        try:
            source = _read_source(fpath)
        except OSError as exc:
            logger.warning("security_surface: read error %s — %s", relname, exc)
            continue
        try:
            fonctions = extract_json(source)["fonctions"]
            tree = ast.parse(source)
        except SyntaxError as exc:
            logger.warning("security_surface: skipping %s — %s", relname, exc)
            continue
        except Exception as exc:  # any other parse failure → skip, never crash
            logger.warning("security_surface: skipping %s — %s", relname, exc)
            continue

        total_files += 1
        node_map = _function_nodes(tree)
        file_had_risk = False

        for fn in fonctions:
            total_functions += 1
            node = node_map.get(fn["nom"])
            high_calls, medium_calls = _scan_function(node, fn["appels"])
            line_hint = f"line {node.lineno}" if node is not None else "unknown"
            if high_calls:
                high_risk.append({
                    "file": relname,
                    "function": fn["nom"],
                    "calls": high_calls,
                    "line_hint": line_hint,
                })
                file_had_risk = True
            elif medium_calls:
                medium_risk.append({
                    "file": relname,
                    "function": fn["nom"],
                    "calls": medium_calls,
                    "line_hint": line_hint,
                })
                file_had_risk = True

        if not file_had_risk:
            clean.append(relname)

    return {
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "clean": sorted(clean),
        "summary": {
            "high_risk_count": len(high_risk),
            "medium_risk_count": len(medium_risk),
            "total_functions_scanned": total_functions,
            "total_files_scanned": total_files,
            "clean_files": len(clean),
        },
    }


# ─────────────────────────────────────────────────────────────────────
# MCP tool definitions + dispatch
# ─────────────────────────────────────────────────────────────────────
_TOOL_SCHEMAS = [
    {
        "name": "index_repo",
        "description": (
            "Walk a repo (Python + JavaScript/TypeScript) and return an "
            "aggregated structural index: every function, class, import and "
            "dependency. Max depth 3; skips __pycache__, .git, venv, node_modules."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo path (relative to --root or absolute).",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_compressed",
        "description": (
            "S5-compress a single source file — Python or JS/TS (signatures + "
            "dependency contracts, bodies stripped) — and return the compressed "
            "structure plus token stats (tokens_before, tokens_after, reduction_pct)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to a .py file (relative to --root or absolute).",
                }
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "expand",
        "description": (
            "Given function/class names in a .py file, return them verbatim "
            "(full source) plus their compressed dependency context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to a .py file (relative to --root or absolute).",
                },
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Function/class names to expand.",
                },
            },
            "required": ["file_path", "targets"],
        },
    },
    {
        "name": "blast_radius",
        "description": (
            "Reverse-call-graph impact analysis for a Python function: BFS over "
            "inverted call edges to find every function transitively affected by "
            "changing target_function. Returns direct_callers, all_impacted, "
            "impacted_count, total_functions and a risk_level. Python files only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to a .py file (relative to --root or absolute).",
                },
                "target_function": {
                    "type": "string",
                    "description": "Function/method name to analyze the blast radius of.",
                },
            },
            "required": ["file_path", "target_function"],
        },
    },
    {
        "name": "security_surface",
        "description": (
            "Walk a Python repo and find functions that call dangerous primitives "
            "— pure AST analysis, zero LLM calls. Classifies calls as HIGH risk "
            "(subprocess/os.system/eval/exec/pickle/__import__/ctypes …) or MEDIUM "
            "risk (sockets/requests/httpx/urllib/paramiko/smtplib and write-mode "
            "open()). Returns high_risk, medium_risk, clean files and a summary. "
            "Max depth 3; skips __pycache__, .git, venv, node_modules."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Repo path (relative to --root or absolute).",
                }
            },
            "required": ["repo_path"],
        },
    },
]


def dispatch(name: str, arguments: dict, root: str) -> dict:
    """Route an MCP tool call to the matching pure function."""
    if name == "index_repo":
        return index_repo(arguments["path"], root=root)
    if name == "get_compressed":
        return get_compressed(arguments["file_path"], root=root)
    if name == "expand":
        return expand(arguments["file_path"], arguments.get("targets", []), root=root)
    if name == "blast_radius":
        return blast_radius(arguments["file_path"], arguments["target_function"], root=root)
    if name == "security_surface":
        return security_surface(arguments["repo_path"], root=root)
    return {"error": f"Unknown tool: {name}"}


# ─────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────
class RefractServer:
    """Local MCP server exposing Refract compression as tools (stdio)."""

    def __init__(self, root: str = ".", verbose: bool = False):
        self.root = os.path.normpath(os.path.abspath(root))
        self.verbose = verbose

    def _build_mcp_server(self):
        try:
            from mcp import types as mt
            from mcp.server import Server
        except ImportError as exc:
            raise RuntimeError(
                f"MCP library missing: {exc}. Run `pip install mcp`"
            ) from exc

        server = Server("refract-code")
        owner = self

        @server.list_tools()
        async def _list_tools():
            return [
                mt.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in _TOOL_SCHEMAS
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict | None = None):
            if owner.verbose:
                print(f"[Refract] → {name}({list((arguments or {}).keys())})", file=sys.stderr)
            try:
                result = dispatch(name, arguments or {}, owner.root)
            except Exception as exc:  # never crash the session
                logger.error("Tool '%s' failed: %s", name, exc)
                result = {"error": f"{type(exc).__name__}: {exc}"}
            return [mt.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        return server

    async def serve(self) -> None:
        """Start the MCP server in stdio mode."""
        try:
            from mcp.server.stdio import stdio_server
        except ImportError as exc:
            raise RuntimeError(
                f"MCP library missing: {exc}. Run `pip install mcp`"
            ) from exc

        server = self._build_mcp_server()
        if self.verbose:
            print(f"[Refract] MCP server started (stdio) — root: {self.root}", file=sys.stderr)

        init_opts = server.create_initialization_options()
        async with stdio_server() as (read, write):
            await server.run(read, write, init_opts)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refract-server",
        description=(
            "Refract — MCP server exposing Python source compression as tools "
            "(index_repo, get_compressed, expand). Runs over stdio for "
            "Claude Desktop / Cursor."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="Root directory paths are resolved against (default: cwd).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log each tool call to stderr.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: WARNING).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    import asyncio

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # keep stdout clean for the MCP protocol
    )

    server = RefractServer(root=args.root, verbose=args.verbose)
    print(f"[Refract] Code server ready — root: {server.root}", file=sys.stderr)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\n[Refract] Server stopped.", file=sys.stderr)
    except Exception as exc:
        print(f"[Refract] Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
