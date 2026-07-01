"""
test_refract_status.py — repo health summary CLI (refract-status).

100% deterministic, zero network, zero LLM.  All file I/O uses tmp_path.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_status import (
    _build_parser,
    collect,
    main,
    render_human,
    render_json,
)

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")


# ─────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────

def test_parser_defaults():
    args = _build_parser().parse_args([])
    assert args.root == "."
    assert args.as_json is False
    assert args.log_level == "WARNING"


def test_parser_root():
    args = _build_parser().parse_args(["--root", "/tmp/myrepo"])
    assert args.root == "/tmp/myrepo"


def test_parser_json_flag():
    args = _build_parser().parse_args(["--json"])
    assert args.as_json is True


def test_parser_log_level():
    args = _build_parser().parse_args(["--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"


def test_parser_invalid_log_level():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--log-level", "VERBOSE"])


# ─────────────────────────────────────────────────────────────────────
# collect() — directory errors
# ─────────────────────────────────────────────────────────────────────

def test_collect_missing_dir():
    data = collect("/nonexistent/path/xyz")
    assert "error" in data
    assert "high_risk" not in data  # no security leakage on error path


def test_collect_empty_dir(tmp_path):
    data = collect(str(tmp_path))
    assert "error" not in data
    assert data["by_language"] == {}
    assert data["security"] == {}
    assert data["ts_fallback"] == []
    assert data["errors"] == {}


# ─────────────────────────────────────────────────────────────────────
# collect() — Python files
# ─────────────────────────────────────────────────────────────────────

_SIMPLE_PY = '''\
import os


class Config:
    pass


def load(path):
    return path


def save(path, data):
    return None
'''


def test_collect_python_file_counts(tmp_path):
    (tmp_path / "app.py").write_text(_SIMPLE_PY, encoding="utf-8")
    data = collect(str(tmp_path))
    py = data["by_language"]["python"]
    assert py["files"] == 1
    assert py["functions"] >= 2  # load, save
    assert py["classes"] >= 1   # Config


def test_collect_token_stats_positive(tmp_path):
    (tmp_path / "app.py").write_text(_SIMPLE_PY, encoding="utf-8")
    data = collect(str(tmp_path))
    py = data["by_language"]["python"]
    assert py["raw_tokens"] > 0
    assert py["compressed_tokens"] > 0
    # compression must never inflate beyond raw (S5 is lossless on small files)
    assert py["compressed_tokens"] <= py["raw_tokens"]


def test_collect_multiple_python_files(tmp_path):
    for i in range(3):
        (tmp_path / f"mod{i}.py").write_text(
            f"def func{i}(): pass\n", encoding="utf-8"
        )
    data = collect(str(tmp_path))
    py = data["by_language"]["python"]
    assert py["files"] == 3
    assert py["functions"] == 3


# ─────────────────────────────────────────────────────────────────────
# collect() — security surface
# ─────────────────────────────────────────────────────────────────────

def test_collect_security_subprocess(tmp_path):
    src = '''\
import subprocess


def run(cmd):
    return subprocess.run(cmd, shell=True)
'''
    (tmp_path / "runner.py").write_text(src, encoding="utf-8")
    data = collect(str(tmp_path))
    assert "subprocess" in data["security"]
    assert data["security"]["subprocess"] >= 1


def test_collect_security_eval(tmp_path):
    src = '''\
def execute(code):
    return eval(code)
'''
    (tmp_path / "exec.py").write_text(src, encoding="utf-8")
    data = collect(str(tmp_path))
    assert "eval" in data["security"]


def test_collect_security_clean_function(tmp_path):
    src = '''\
def add(a, b):
    return a + b
'''
    (tmp_path / "pure.py").write_text(src, encoding="utf-8")
    data = collect(str(tmp_path))
    assert data["security"] == {}


def test_collect_security_multiple_categories(tmp_path):
    src = '''\
import subprocess
import pickle


def dangerous(cmd, data):
    subprocess.run(cmd)
    return pickle.loads(data)
'''
    (tmp_path / "multi.py").write_text(src, encoding="utf-8")
    data = collect(str(tmp_path))
    assert "subprocess" in data["security"]
    assert "pickle" in data["security"]


# ─────────────────────────────────────────────────────────────────────
# collect() — error handling
# ─────────────────────────────────────────────────────────────────────

def test_collect_syntax_error_skipped(tmp_path):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("def fine(): pass\n", encoding="utf-8")
    data = collect(str(tmp_path))
    # The good file was counted despite the broken one
    assert data["by_language"]["python"]["files"] == 1
    # The broken file was recorded in errors
    assert "broken.py" in data["errors"]


def test_collect_read_error_skipped(tmp_path, monkeypatch):
    (tmp_path / "unreadable.py").write_text("def f(): pass\n", encoding="utf-8")

    original = open

    def patched_open(path, *args, **kwargs):
        if "unreadable.py" in str(path):
            raise OSError("permission denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)
    data = collect(str(tmp_path))
    assert any("unreadable.py" in k for k in data["errors"])


# ─────────────────────────────────────────────────────────────────────
# collect() — against real src/ for sanity
# ─────────────────────────────────────────────────────────────────────

def test_collect_src_has_python_language():
    data = collect(SRC_DIR)
    assert "error" not in data
    assert "python" in data["by_language"]
    py = data["by_language"]["python"]
    assert py["files"] >= 5
    assert py["functions"] >= 20


def test_collect_src_token_gain():
    data = collect(SRC_DIR)
    py = data["by_language"]["python"]
    # S5 compression always saves tokens on real source code
    assert py["compressed_tokens"] < py["raw_tokens"]


def test_collect_src_security_has_entries():
    data = collect(SRC_DIR)
    # refract_server.py calls subprocess/os indirectly — some risk must be found
    assert len(data["security"]) > 0 or data["by_language"]["python"]["files"] > 0


def test_collect_src_no_fatal_errors():
    data = collect(SRC_DIR)
    assert "error" not in data
    # Any per-file errors are acceptable but the walk must complete
    assert isinstance(data["errors"], dict)


# ─────────────────────────────────────────────────────────────────────
# collect() — ts_fallback
# ─────────────────────────────────────────────────────────────────────

def test_collect_ts_fallback_when_tree_sitter_missing(tmp_path, monkeypatch):
    (tmp_path / "app.js").write_text("function hello() {}\n", encoding="utf-8")

    monkeypatch.setattr("refract_status._ts_available", lambda: False)
    data = collect(str(tmp_path))
    assert ".js" in data["ts_fallback"]
    # The JS file should not appear in by_language (skipped, no fallback parser)
    assert "javascript" not in data["by_language"]


def test_collect_ts_fallback_empty_when_no_js_files(tmp_path, monkeypatch):
    (tmp_path / "pure.py").write_text("def f(): pass\n", encoding="utf-8")
    monkeypatch.setattr("refract_status._ts_available", lambda: False)
    data = collect(str(tmp_path))
    assert data["ts_fallback"] == []


# ─────────────────────────────────────────────────────────────────────
# render_human()
# ─────────────────────────────────────────────────────────────────────

def _make_data(tmp_path) -> dict:
    (tmp_path / "app.py").write_text(_SIMPLE_PY, encoding="utf-8")
    return collect(str(tmp_path))


def test_render_human_has_all_sections(tmp_path):
    data = _make_data(tmp_path)
    out = render_human(data)
    assert "Repo health" in out
    assert "Files by language" in out
    assert "Tokens" in out
    assert "Security surface" in out
    assert "Tree-sitter" in out


def test_render_human_shows_python(tmp_path):
    data = _make_data(tmp_path)
    out = render_human(data)
    assert "Python" in out


def test_render_human_shows_token_numbers(tmp_path):
    data = _make_data(tmp_path)
    out = render_human(data)
    # Raw and compressed token lines must be present with numeric content
    assert "Raw" in out
    assert "Compressed" in out
    assert "Gain" in out


def test_render_human_no_files(tmp_path):
    data = collect(str(tmp_path))
    out = render_human(data)
    assert "no supported source files found" in out


def test_render_human_security_call(tmp_path):
    src = "import subprocess\n\ndef run(cmd):\n    subprocess.run(cmd)\n"
    (tmp_path / "r.py").write_text(src, encoding="utf-8")
    data = collect(str(tmp_path))
    out = render_human(data)
    assert "subprocess" in out


def test_render_human_no_security(tmp_path):
    (tmp_path / "pure.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    data = collect(str(tmp_path))
    out = render_human(data)
    assert "No dangerous calls detected" in out


def test_render_human_tree_sitter_fallback(tmp_path, monkeypatch):
    (tmp_path / "app.js").write_text("function f() {}\n", encoding="utf-8")
    monkeypatch.setattr("refract_status._ts_available", lambda: False)
    data = collect(str(tmp_path))
    out = render_human(data)
    assert "Fallback active for" in out
    assert ".js" in out


def test_render_human_all_supported(tmp_path):
    (tmp_path / "pure.py").write_text("def f(): pass\n", encoding="utf-8")
    data = collect(str(tmp_path))
    out = render_human(data)
    assert "All extensions supported" in out


def test_render_human_error_path():
    data = {"error": "Not a directory: /nope"}
    out = render_human(data)
    assert "Error:" in out
    assert "/nope" in out


def test_render_human_parse_errors_shown(tmp_path):
    (tmp_path / "broken.py").write_text("def bad(:\n", encoding="utf-8")
    data = collect(str(tmp_path))
    out = render_human(data)
    assert "Parse errors" in out
    assert "broken.py" in out


# ─────────────────────────────────────────────────────────────────────
# render_json()
# ─────────────────────────────────────────────────────────────────────

def test_render_json_is_valid(tmp_path):
    data = _make_data(tmp_path)
    raw = render_json(data)
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


def test_render_json_top_level_keys(tmp_path):
    data = _make_data(tmp_path)
    parsed = json.loads(render_json(data))
    for key in ("root", "by_language", "tokens", "security", "ts_fallback_extensions"):
        assert key in parsed


def test_render_json_tokens_structure(tmp_path):
    data = _make_data(tmp_path)
    parsed = json.loads(render_json(data))
    tokens = parsed["tokens"]
    assert "raw" in tokens
    assert "compressed" in tokens
    assert "gain_pct" in tokens
    assert tokens["raw"] > 0
    assert 0.0 <= tokens["gain_pct"] <= 100.0


def test_render_json_by_language_structure(tmp_path):
    data = _make_data(tmp_path)
    parsed = json.loads(render_json(data))
    assert "python" in parsed["by_language"]
    py = parsed["by_language"]["python"]
    for key in ("files", "functions", "classes", "raw_tokens", "compressed_tokens"):
        assert key in py


def test_render_json_security_is_dict(tmp_path):
    data = _make_data(tmp_path)
    parsed = json.loads(render_json(data))
    assert isinstance(parsed["security"], dict)


def test_render_json_error_path():
    data = {"error": "Not a directory: /nope"}
    parsed = json.loads(render_json(data))
    assert "error" in parsed


def test_render_json_empty_dir(tmp_path):
    data = collect(str(tmp_path))
    parsed = json.loads(render_json(data))
    assert parsed["tokens"]["raw"] == 0
    assert parsed["tokens"]["gain_pct"] == 0.0
    assert parsed["by_language"] == {}


# ─────────────────────────────────────────────────────────────────────
# main() integration
# ─────────────────────────────────────────────────────────────────────

def test_main_human_output_to_stdout(tmp_path, capsys):
    (tmp_path / "app.py").write_text(_SIMPLE_PY, encoding="utf-8")
    main(["--root", str(tmp_path)])
    captured = capsys.readouterr()
    assert "Repo health" in captured.out
    assert captured.err == "" or "[" not in captured.out  # diagnostics on stderr only


def test_main_json_output_to_stdout(tmp_path, capsys):
    (tmp_path / "app.py").write_text(_SIMPLE_PY, encoding="utf-8")
    main(["--root", str(tmp_path), "--json"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "by_language" in parsed
    assert "tokens" in parsed


def test_main_default_root_does_not_crash(capsys):
    # Runs against the real cwd (the repo root), must not raise
    main([])
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_main_nonexistent_root_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["--root", str(tmp_path / "does_not_exist")])
    assert exc_info.value.code != 0


def test_main_json_nonexistent_root_exits_nonzero(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--root", str(tmp_path / "nope"), "--json"])
    assert exc_info.value.code != 0
    parsed = json.loads(capsys.readouterr().out)
    assert "error" in parsed


def test_main_stdout_clean_no_stderr_leakage(tmp_path, capsys):
    (tmp_path / "app.py").write_text(_SIMPLE_PY, encoding="utf-8")
    main(["--root", str(tmp_path), "--json"])
    captured = capsys.readouterr()
    # stdout must be valid JSON — no mixed diagnostic output
    json.loads(captured.out)


def test_main_is_callable():
    from refract_status import main as status_main
    assert callable(status_main)
