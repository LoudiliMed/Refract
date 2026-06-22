"""
test_blast_radius.py — reverse-call-graph impact analysis (blast_radius tool).

Deterministic: no subprocess, no network.
"""

from __future__ import annotations

import os
import sys

import pytest

# Access src/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_server import blast_radius, dispatch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Known call graph: A calls B, C calls A, D calls A.
#   reverse edges → B is called by A; A is called by C and D.
#   blast radius of B = {A, C, D}.
CALL_GRAPH = '''\
def A():
    return B()


def B():
    return 1


def C():
    return A()


def D():
    return A()
'''


@pytest.fixture
def graph_file(tmp_path):
    f = tmp_path / "graph.py"
    f.write_text(CALL_GRAPH, encoding="utf-8")
    return str(f)


# ─── Test 1: basic blast radius ───────────────────────────────────────────────

def test_basic_blast_radius(graph_file):
    result = blast_radius(graph_file, "B")
    assert result["target"] == "B"
    assert result["direct_callers"] == ["A"]
    assert result["all_impacted"] == ["A", "C", "D"]
    assert result["impacted_count"] == 3
    assert result["total_functions"] == 4
    assert result["risk_level"] == "MEDIUM"  # 3 > 2


def test_transitive_only_one_hop(graph_file):
    # A is called directly by C and D; B is not a caller of A.
    result = blast_radius(graph_file, "A")
    assert result["direct_callers"] == ["C", "D"]
    assert result["all_impacted"] == ["C", "D"]
    assert result["impacted_count"] == 2
    assert result["risk_level"] == "LOW"  # 2 is not > 2


# ─── Test 2: function with no callers ─────────────────────────────────────────

def test_leaf_function_has_empty_radius(graph_file):
    # Nobody calls D → empty blast radius.
    result = blast_radius(graph_file, "D")
    assert result["target"] == "D"
    assert result["direct_callers"] == []
    assert result["all_impacted"] == []
    assert result["impacted_count"] == 0
    assert result["risk_level"] == "LOW"


# ─── Test 3: function not found ───────────────────────────────────────────────

def test_function_not_found_lists_available(graph_file):
    result = blast_radius(graph_file, "nonexistent_function")
    assert "error" in result
    assert "nonexistent_function" in result["error"]
    assert result["available_functions"] == ["A", "B", "C", "D"]
    # A clear error, not a partial/blast result
    assert "all_impacted" not in result


# ─── Test 4: run on a real file ───────────────────────────────────────────────

def test_real_file_returns_valid_dict():
    target = os.path.join(REPO_ROOT, "src", "refract_proxy.py")
    result = blast_radius(target, "handle_tools_list")
    assert "error" not in result
    expected_keys = {
        "target",
        "direct_callers",
        "all_impacted",
        "impacted_count",
        "total_functions",
        "risk_level",
    }
    assert expected_keys <= set(result)
    assert result["target"] == "handle_tools_list"
    assert isinstance(result["direct_callers"], list)
    assert isinstance(result["all_impacted"], list)
    assert result["impacted_count"] == len(result["all_impacted"])
    assert result["total_functions"] > 0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


# ─── class methods are part of the graph ──────────────────────────────────────

def test_includes_class_methods(tmp_path):
    src = '''\
def helper():
    return 1


class Service:
    def run(self):
        return helper()

    def start(self):
        return helper()
'''
    f = tmp_path / "svc.py"
    f.write_text(src, encoding="utf-8")
    result = blast_radius(str(f), "helper")
    # Both methods call helper() by bare name → both impacted.
    assert result["target"] == "helper"
    assert set(result["all_impacted"]) == {"run", "start"}
    # helper, run, start
    assert result["total_functions"] == 3


# ─── risk thresholds ──────────────────────────────────────────────────────────

def test_high_risk_threshold(tmp_path):
    # One core function called by 6 others → impacted_count 6 > 5 → HIGH.
    callers = "\n\n".join(f"def caller_{i}():\n    return core()" for i in range(6))
    src = f"def core():\n    return 1\n\n\n{callers}\n"
    f = tmp_path / "hot.py"
    f.write_text(src, encoding="utf-8")
    result = blast_radius(str(f), "core")
    assert result["impacted_count"] == 6
    assert result["risk_level"] == "HIGH"


# ─── robustness ───────────────────────────────────────────────────────────────

def test_syntax_error_returns_error(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def broken(:\n", encoding="utf-8")
    result = blast_radius(str(f), "broken")
    assert "error" in result


def test_missing_file_returns_error():
    result = blast_radius("/nonexistent/file.py", "foo")
    assert "error" in result


# ─── dispatch wiring ──────────────────────────────────────────────────────────

def test_dispatch_routes_blast_radius(graph_file, tmp_path):
    result = dispatch(
        "blast_radius",
        {"file_path": "graph.py", "target_function": "B"},
        root=str(tmp_path),
    )
    assert result["all_impacted"] == ["A", "C", "D"]
