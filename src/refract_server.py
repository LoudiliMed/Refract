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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast_extractor
from ast_extractor import IndexModule, compress, count_tokens, dependances, extract

logger = logging.getLogger(__name__)

# Directories never worth walking into.
_SKIP_DIRS = frozenset({"__pycache__", ".git", "venv", ".venv", "node_modules"})

# Walk no deeper than this many directory levels below the root.
_MAX_DEPTH = 3


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
def _iter_py_files(root: str, max_depth: int = _MAX_DEPTH):
    """Yield .py files under *root*, skipping noise dirs, capped at *max_depth*.

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
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _index_file(source: str) -> dict:
    """Per-file index: function names, class names, imports."""
    tree = ast.parse(source)
    idx = IndexModule(tree)
    return {
        "functions": sorted(idx.fonctions),
        "classes": sorted(idx.classes),
        "imports": sorted(idx.imports),
    }


def index_repo(path: str, root: str = ".", max_depth: int = _MAX_DEPTH) -> dict:
    """Walk a Python repo and return an aggregated structural index.

    Runs the AST extractor on every ``.py`` file (max depth 3, skipping
    ``__pycache__`` / ``.git`` / ``venv`` / ``node_modules``) and aggregates
    every function, class, import and external dependency.
    """
    base = _resolve(path, root)
    if not os.path.isdir(base):
        return {"error": f"Not a directory: {base}", "root": base}

    files: dict[str, dict] = {}
    errors: dict[str, str] = {}
    dependencies: set[str] = set()
    n_functions = n_classes = 0

    for fpath in _iter_py_files(base, max_depth):
        relname = os.path.relpath(fpath, base)
        try:
            info = _index_file(_read_source(fpath))
        except SyntaxError as exc:
            errors[relname] = f"syntax error: {exc}"
            continue
        except OSError as exc:
            errors[relname] = f"read error: {exc}"
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
    """S5-compress a single ``.py`` file and report token savings."""
    target = _resolve(file_path, root)
    try:
        source = _read_source(target)
    except OSError as exc:
        return {"error": f"read error: {exc}", "file": target}

    try:
        compressed = compress(source)
    except SyntaxError as exc:
        return {"error": f"syntax error: {exc}", "file": target}

    tokens_before = count_tokens(source)
    tokens_after = count_tokens(compressed)
    reduction = (
        round((1 - tokens_after / tokens_before) * 100, 1) if tokens_before else 0.0
    )
    return {
        "file": target,
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

    For each name in *targets* found at module level (or as a top-level method
    inside a class), returns the full source plus the compressed dependency
    contract (data / type / internal / external) computed against the module
    vocabulary.
    """
    target_path = _resolve(file_path, root)
    try:
        source = _read_source(target_path)
    except OSError as exc:
        return {"error": f"read error: {exc}", "file": target_path}

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
# MCP tool definitions + dispatch
# ─────────────────────────────────────────────────────────────────────
_TOOL_SCHEMAS = [
    {
        "name": "index_repo",
        "description": (
            "Walk a Python repo and return an aggregated structural index: "
            "every function, class, import and external dependency. Max depth 3; "
            "skips __pycache__, .git, venv, node_modules."
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
            "S5-compress a single .py file (signatures + dependency contracts, "
            "bodies stripped) and return the compressed structure plus token "
            "stats (tokens_before, tokens_after, reduction_pct)."
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
]


def dispatch(name: str, arguments: dict, root: str) -> dict:
    """Route an MCP tool call to the matching pure function."""
    if name == "index_repo":
        return index_repo(arguments["path"], root=root)
    if name == "get_compressed":
        return get_compressed(arguments["file_path"], root=root)
    if name == "expand":
        return expand(arguments["file_path"], arguments.get("targets", []), root=root)
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
