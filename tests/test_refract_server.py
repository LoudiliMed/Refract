"""
test_refract_server.py — Verifies the refract_server MCP tools:
  - index_repo aggregates functions/classes/imports, respects depth + skip dirs
  - get_compressed compresses a file and reports token savings
  - expand returns named defs verbatim + their dependency context
  - dispatch routes by tool name; CLI parser exposes --root
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

# Access src/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import refract_server
from refract_server import (
    RefractServer,
    _build_parser,
    _iter_py_files,
    dispatch,
    expand,
    get_compressed,
    index_repo,
)

# ─── sample source ───────────────────────────────────────────────────────────

SAMPLE = '''\
import os
import json

CONFIG = {"k": 1}


def helper(x):
    """Doubles x."""
    return x * 2


def run(payload):
    """Orchestrates the run."""
    data = json.dumps(payload)
    n = helper(CONFIG["k"])
    return os.path.join(data, str(n))


class Worker:
    """A worker."""

    def __init__(self):
        self.count = 0

    def tick(self):
        self.count += 1
        return self.count
'''


@pytest.fixture
def repo(tmp_path):
    """A small repo with nested dirs + noise dirs to be skipped."""
    (tmp_path / "mod_a.py").write_text(SAMPLE, encoding="utf-8")

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod_b.py").write_text("import sys\n\ndef b():\n    return sys.argv\n", encoding="utf-8")

    # Noise dir that must be skipped
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.py").write_text("def ignored():\n    return 1\n", encoding="utf-8")

    # Too deep (level 3) — beyond max depth
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "too_deep.py").write_text("def too_deep():\n    return 1\n", encoding="utf-8")

    return tmp_path


# ─── index_repo ──────────────────────────────────────────────────────────────

def test_index_repo_aggregates_defs(repo):
    result = index_repo(str(repo))
    assert result["totals"]["files"] == 2  # mod_a.py + pkg/mod_b.py
    assert "mod_a.py" in result["files"]
    assert "pkg/mod_b.py" in result["files"]

    mod_a = result["files"]["mod_a.py"]
    assert "helper" in mod_a["functions"]
    assert "run" in mod_a["functions"]
    assert "Worker" in mod_a["classes"]
    assert {"os", "json"} <= set(mod_a["imports"])


def test_index_repo_dependencies_union(repo):
    result = index_repo(str(repo))
    assert {"os", "json", "sys"} <= set(result["dependencies"])
    assert result["totals"]["dependencies"] == len(result["dependencies"])


def test_index_repo_skips_noise_dirs(repo):
    result = index_repo(str(repo))
    assert all("__pycache__" not in f for f in result["files"])


def test_index_repo_respects_max_depth(repo):
    # a/b/c/too_deep.py is at depth 3 → excluded
    result = index_repo(str(repo))
    assert all("too_deep" not in f for f in result["files"])


def test_index_repo_bad_path():
    result = index_repo("/nonexistent/path/xyz")
    assert "error" in result


def test_iter_py_files_depth_cap(repo):
    files = list(_iter_py_files(str(repo)))
    names = {os.path.basename(f) for f in files}
    assert "mod_a.py" in names
    assert "mod_b.py" in names
    assert "too_deep.py" not in names
    assert "ignored.py" not in names


def test_index_repo_records_syntax_errors(tmp_path):
    (tmp_path / "ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    result = index_repo(str(tmp_path))
    assert "ok.py" in result["files"]
    assert "broken.py" in result.get("errors", {})


# ─── get_compressed ──────────────────────────────────────────────────────────

def test_get_compressed_reports_token_savings(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(SAMPLE, encoding="utf-8")
    result = get_compressed(str(f))
    assert result["tokens_before"] > 0
    assert result["tokens_after"] > 0
    assert "compressed" in result
    # Signatures survive compression
    assert "def run(" in result["compressed"]
    # reduction_pct is the honest formula (may be ~0 on tiny files)
    expected = round((1 - result["tokens_after"] / result["tokens_before"]) * 100, 1)
    assert result["reduction_pct"] == expected


def test_get_compressed_shrinks_heavy_bodies(tmp_path):
    # A TOOL function with a long body: S5 strips it → real reduction.
    body = "\n".join(f"    step_{i} = compute(payload, {i}) + step_{i - 1}" for i in range(1, 40))
    src = f"import compute\n\nROOT = 0\n\n\ndef heavy(payload):\n    step_0 = ROOT\n{body}\n    return step_39\n"
    f = tmp_path / "heavy.py"
    f.write_text(src, encoding="utf-8")
    result = get_compressed(str(f))
    assert result["tokens_after"] < result["tokens_before"]
    assert result["reduction_pct"] > 0


def test_get_compressed_bad_file():
    result = get_compressed("/nonexistent/file.py")
    assert "error" in result


def test_get_compressed_syntax_error(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def broken(:\n", encoding="utf-8")
    result = get_compressed(str(f))
    assert "error" in result


# ─── expand ──────────────────────────────────────────────────────────────────

def test_expand_returns_verbatim_source(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(SAMPLE, encoding="utf-8")
    result = expand(str(f), ["run", "Worker"])

    assert set(result["targets"]) == {"run", "Worker"}
    assert result["missing"] == []

    run = result["targets"]["run"]
    assert run["kind"] == "function"
    assert 'data = json.dumps(payload)' in run["source"]
    # Dependency context: helper is an internal call, json/os external
    deps = run["dependencies"]
    assert "helper" in deps["interne"]
    assert "json" in deps["externe"] or "os" in deps["externe"]
    assert "CONFIG" in deps["data"]

    worker = result["targets"]["Worker"]
    assert worker["kind"] == "class"
    assert "def tick(self)" in worker["source"]


def test_expand_reports_missing(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(SAMPLE, encoding="utf-8")
    result = expand(str(f), ["run", "does_not_exist"])
    assert "run" in result["targets"]
    assert result["missing"] == ["does_not_exist"]


def test_expand_context_imports(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(SAMPLE, encoding="utf-8")
    result = expand(str(f), ["helper"])
    assert {"os", "json"} <= set(result["context"]["imports"])


def test_expand_bad_file():
    result = expand("/nonexistent/file.py", ["foo"])
    assert "error" in result


# ─── path resolution against root ─────────────────────────────────────────────

def test_relative_paths_resolve_against_root(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(SAMPLE, encoding="utf-8")
    # file_path relative, resolved against root
    result = get_compressed("mod.py", root=str(tmp_path))
    assert result["tokens_before"] > 0
    assert result["file"].endswith("mod.py")


# ─── dispatch ─────────────────────────────────────────────────────────────────

def test_dispatch_routes_each_tool(repo):
    idx = dispatch("index_repo", {"path": str(repo)}, root=str(repo))
    assert idx["totals"]["files"] == 2

    comp = dispatch("get_compressed", {"file_path": "mod_a.py"}, root=str(repo))
    assert comp["tokens_before"] > 0

    exp = dispatch("expand", {"file_path": "mod_a.py", "targets": ["run"]}, root=str(repo))
    assert "run" in exp["targets"]


def test_dispatch_unknown_tool():
    result = dispatch("nope", {}, root=".")
    assert "error" in result


# ─── MCP server wiring ────────────────────────────────────────────────────────

def test_server_lists_three_tools(tmp_path):
    pytest.importorskip("mcp")
    server = RefractServer(root=str(tmp_path))
    mcp_server = server._build_mcp_server()

    handler = mcp_server.request_handlers
    # The list_tools handler is registered; exercise it via the decorated coroutine.
    # Pull tool names from the static schema the server advertises.
    names = {t["name"] for t in refract_server._TOOL_SCHEMAS}
    assert names == {"index_repo", "get_compressed", "expand"}


def test_server_call_tool_returns_json_text(tmp_path):
    mt = pytest.importorskip("mcp.types")
    f = tmp_path / "mod.py"
    f.write_text(SAMPLE, encoding="utf-8")

    server = RefractServer(root=str(tmp_path))
    mcp_server = server._build_mcp_server()

    # Find the registered call_tool handler and invoke it.
    from mcp import types as mtypes

    call_handler = mcp_server.request_handlers[mtypes.CallToolRequest]
    req = mtypes.CallToolRequest(
        method="tools/call",
        params=mtypes.CallToolRequestParams(
            name="get_compressed", arguments={"file_path": "mod.py"}
        ),
    )
    result = asyncio.run(call_handler(req))
    # Unwrap ServerResult → CallToolResult → content
    content = result.root.content
    assert content and content[0].type == "text"
    payload = json.loads(content[0].text)
    assert payload["tokens_before"] > 0


# ─── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_parser_exposes_root():
    args = _build_parser().parse_args(["--root", "/tmp/repo", "--verbose"])
    assert args.root == "/tmp/repo"
    assert args.verbose is True


def test_cli_parser_defaults():
    args = _build_parser().parse_args([])
    assert args.root == "."
    assert args.verbose is False
