"""
test_security_surface.py — dangerous-primitive scanner (security_surface tool).

Pure AST analysis, zero LLM calls. Deterministic: no subprocess, no network.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

# Access src/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_server import dispatch, security_surface

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")


def _find(entries, function):
    """Return the first risk entry for *function*, or None."""
    return next((e for e in entries if e["function"] == function), None)


# ─── Test 1: high risk detection ──────────────────────────────────────────────

def test_high_risk_detection(tmp_path):
    src = '''\
import subprocess


def run_command(cmd):
    return subprocess.run(cmd, shell=True)
'''
    f = tmp_path / "danger.py"
    f.write_text(src, encoding="utf-8")

    result = security_surface(str(tmp_path))

    entry = _find(result["high_risk"], "run_command")
    assert entry is not None
    assert entry["file"] == "danger.py"
    assert "subprocess.run" in entry["calls"]
    assert "line" in entry["line_hint"]
    # A high-risk function is not also listed under medium_risk.
    assert _find(result["medium_risk"], "run_command") is None


# ─── Test 2: medium risk detection ────────────────────────────────────────────

def test_medium_risk_detection(tmp_path):
    src = '''\
import requests


def fetch(url):
    return requests.get(url)
'''
    f = tmp_path / "net.py"
    f.write_text(src, encoding="utf-8")

    result = security_surface(str(tmp_path))

    entry = _find(result["medium_risk"], "fetch")
    assert entry is not None
    assert entry["file"] == "net.py"
    assert "requests.get" in entry["calls"]
    assert _find(result["high_risk"], "fetch") is None


# ─── Test 3: clean file ───────────────────────────────────────────────────────

def test_clean_file():
    # ast_extractor.py performs only pure AST work — no dangerous primitives.
    result = security_surface(SRC_DIR)
    assert "ast_extractor.py" in result["clean"]
    # ...and it appears in no risk list.
    for entry in result["high_risk"] + result["medium_risk"]:
        assert entry["file"] != "ast_extractor.py"


# ─── Test 4: mixed repo (full src/) ───────────────────────────────────────────

def test_mixed_repo_structure_and_consistency():
    result = security_surface(SRC_DIR)

    # All expected top-level keys present.
    for key in ("high_risk", "medium_risk", "clean", "summary"):
        assert key in result
    summary = result["summary"]
    for key in (
        "high_risk_count",
        "medium_risk_count",
        "total_functions_scanned",
        "total_files_scanned",
        "clean_files",
    ):
        assert key in summary

    # Summary counts mirror the lists.
    assert summary["high_risk_count"] == len(result["high_risk"])
    assert summary["medium_risk_count"] == len(result["medium_risk"])
    assert summary["clean_files"] == len(result["clean"])

    # File partition: every scanned file is either clean or has >=1 risk entry.
    risky_files = {e["file"] for e in result["high_risk"]}
    risky_files |= {e["file"] for e in result["medium_risk"]}
    assert len(risky_files) + summary["clean_files"] == summary["total_files_scanned"]


# ─── Test 5: missing repo path ────────────────────────────────────────────────

def test_missing_repo_path_returns_error():
    result = security_surface("/nonexistent/repo/path")
    assert "error" in result
    # Never raises, and no risk keys leak through on the error path.
    assert "high_risk" not in result


# ─── Test 6: file with syntax error skipped gracefully ────────────────────────

def test_syntax_error_skipped_and_logged(tmp_path, caplog):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text(
        "import os\n\n\ndef wipe():\n    os.system('rm -rf /')\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        result = security_surface(str(tmp_path))

    # The good file was still scanned despite the broken one.
    assert result["summary"]["total_files_scanned"] == 1
    assert _find(result["high_risk"], "wipe") is not None
    # The broken file appears nowhere.
    assert "broken.py" not in result["clean"]
    # Skip was logged (to stderr via the module logger).
    assert any("broken.py" in rec.getMessage() for rec in caplog.records)


# ─── Test 7: summary consistency ──────────────────────────────────────────────

def test_summary_function_count_invariant():
    result = security_surface(SRC_DIR)
    summary = result["summary"]
    # Each function lands in at most one risk list, so entries never exceed
    # the number of functions scanned.
    assert summary["total_functions_scanned"] >= (
        summary["high_risk_count"] + summary["medium_risk_count"]
    )


# ─── open() mode sensitivity ──────────────────────────────────────────────────

def test_open_write_flagged_read_ignored(tmp_path):
    src = '''\
def reader(p):
    with open(p) as f:
        return f.read()


def writer(p, data):
    with open(p, "w") as f:
        f.write(data)
'''
    f = tmp_path / "io.py"
    f.write_text(src, encoding="utf-8")

    result = security_surface(str(tmp_path))

    writer_entry = _find(result["medium_risk"], "writer")
    assert writer_entry is not None
    assert "open" in writer_entry["calls"]
    # Read-only open() is not a risk.
    assert _find(result["medium_risk"], "reader") is None


# ─── dispatch wiring ──────────────────────────────────────────────────────────

def test_dispatch_routes_security_surface(tmp_path):
    (tmp_path / "x.py").write_text(
        "import pickle\n\n\ndef load(b):\n    return pickle.loads(b)\n",
        encoding="utf-8",
    )
    result = dispatch("security_surface", {"repo_path": "."}, root=str(tmp_path))
    entry = _find(result["high_risk"], "load")
    assert entry is not None
    assert "pickle.loads" in entry["calls"]
