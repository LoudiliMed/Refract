"""
test_ts_extractor.py — JavaScript / TypeScript support via tree-sitter.

Covers:
  - extract / compress / count_tokens on real JS and TS fixtures (50+ lines)
  - expand: verbatim defs + dependency context
  - language auto-detection wired through refract_server (.js / .ts / .jsx / .tsx)
  - graceful fallback shape on parse-able-but-broken input

All deterministic: no subprocess, no network.
"""

from __future__ import annotations

import os
import sys

import pytest

# Access src/ modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import refract_server
import ts_extractor as ts
from refract_server import _detect, expand, get_compressed, index_repo

SCHEMAS = os.path.join(os.path.dirname(__file__), "..", "schemas")
JS_FIXTURE = os.path.join(SCHEMAS, "sample_app.js")
TS_FIXTURE = os.path.join(SCHEMAS, "sample_app.ts")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def js_source() -> str:
    src = _read(JS_FIXTURE)
    assert src.count("\n") >= 50, "JS fixture must be 50+ lines"
    return src


@pytest.fixture(scope="module")
def ts_source() -> str:
    src = _read(TS_FIXTURE)
    assert src.count("\n") >= 50, "TS fixture must be 50+ lines"
    return src


# ─── JS: extract ──────────────────────────────────────────────────────────────

def test_js_extract_imports(js_source):
    result = ts.extract(js_source, "javascript")
    # import … from "x"  AND  require("x")
    assert {"http", "events", "timers/promises", "crypto"} <= set(result["imports"])


def test_js_extract_classes(js_source):
    result = ts.extract(js_source, "javascript")
    assert {"Task", "Queue"} <= set(result["classes"])


def test_js_extract_functions(js_source):
    result = ts.extract(js_source, "javascript")
    names = {f["nom"] for f in result["fonctions"]}
    # top-level functions, arrow consts, and class methods all captured
    assert {"fetchJson", "request", "uid", "backoff"} <= names
    assert "isRetryable" in names          # const arrow = (...) => ...
    assert {"run", "describe", "add", "drain"} <= names  # methods


def test_js_extract_params_and_internal_calls(js_source):
    result = ts.extract(js_source, "javascript")
    by_name = {f["nom"]: f for f in result["fonctions"]}
    assert by_name["fetchJson"]["params"] == ["url", "opts"]
    # fetchJson calls request() and backoff() — both module functions
    assert "request" in by_name["fetchJson"]["appels"]
    assert "backoff" in by_name["fetchJson"]["appels"]


# ─── JS: compress ─────────────────────────────────────────────────────────────

def test_js_compress_reduces_tokens(js_source):
    compressed = ts.compress(js_source, "javascript")
    before = ts.count_tokens(js_source)
    after = ts.count_tokens(compressed)
    assert after < before
    # On a real ~130-line file the win is substantial.
    assert (1 - after / before) > 0.4


def test_js_compress_keeps_signatures_strips_bodies(js_source):
    compressed = ts.compress(js_source, "javascript")
    # Signatures + imports + class/method names kept
    assert "function fetchJson(url, opts = {})" in compressed
    assert 'import http from "http";' in compressed
    assert "class Task {" in compressed
    assert "describe()" in compressed
    # Bodies stripped
    assert "{ ... }" in compressed
    assert "crypto.randomBytes" not in compressed  # uid() body gone
    # Data vocabulary preserved verbatim
    assert "retries: 3" in compressed


# ─── JS: expand ───────────────────────────────────────────────────────────────

def test_js_expand_verbatim_and_deps(js_source):
    result = ts.expand(js_source, ["fetchJson", "Queue", "ghost"], "javascript")
    assert set(result["targets"]) == {"fetchJson", "Queue"}
    assert result["missing"] == ["ghost"]

    fj = result["targets"]["fetchJson"]
    assert fj["kind"] == "function"
    assert "for (let attempt" in fj["source"]          # full body verbatim
    assert "request" in fj["dependencies"]["interne"]  # internal call
    assert "sleep" in fj["dependencies"]["externe"]    # imported binding used

    q = result["targets"]["Queue"]
    assert q["kind"] == "class"
    assert "drain()" in q["source"]


# ─── TS: extract + compress ───────────────────────────────────────────────────

def test_ts_extract(ts_source):
    result = ts.extract(ts_source, "typescript")
    assert {"events", "./logger"} <= set(result["imports"])
    assert {"Worker", "EchoWorker"} <= set(result["classes"])
    names = {f["nom"] for f in result["fonctions"]}
    assert {"runHandler", "nextId", "process", "handle"} <= names


def test_ts_typed_params_captured(ts_source):
    result = ts.extract(ts_source, "typescript")
    by_name = {f["nom"]: f for f in result["fonctions"]}
    # TS optional/typed params reduce to their binding names
    assert by_name["nextId"]["params"] == ["prefix"]
    assert by_name["runHandler"]["params"] == ["handler", "payload"]


def test_ts_compress_keeps_types_and_return_annotations(ts_source):
    compressed = ts.compress(ts_source, "typescript")
    assert ts.count_tokens(compressed) < ts.count_tokens(ts_source)
    # Interfaces / type aliases kept verbatim (pure vocabulary)
    assert "interface Job<T>" in compressed
    assert "type Handler<T, R>" in compressed
    # Return type annotation survives in the stripped signature
    assert "): Promise<Result<R>> { ... }" in compressed
    # Abstract method signature kept (it has no body to strip)
    assert "abstract handle(job: Job<T>): Promise<R>" in compressed


# ─── language detection in refract_server ──────────────────────────────────────

@pytest.mark.parametrize(
    "name,backend,language",
    [
        ("a.py", "python", None),
        ("a.js", "ts", "javascript"),
        ("a.jsx", "ts", "javascript"),
        ("a.ts", "ts", "typescript"),
        ("a.tsx", "ts", "tsx"),
        ("a.mjs", "ts", "javascript"),
    ],
)
def test_detect_extension(name, backend, language):
    assert _detect(name) == (backend, language)


def test_server_get_compressed_autodetects_js():
    result = get_compressed(JS_FIXTURE)
    assert result["language"] == "javascript"
    assert result["tokens_after"] < result["tokens_before"]
    assert result["reduction_pct"] > 0


def test_server_get_compressed_autodetects_ts():
    result = get_compressed(TS_FIXTURE)
    assert result["language"] == "typescript"
    assert "interface Job<T>" in result["compressed"]


def test_server_expand_autodetects_js():
    result = expand(JS_FIXTURE, ["Task"])
    assert "Task" in result["targets"]
    assert result["targets"]["Task"]["kind"] == "class"
    assert "events" in result["context"]["imports"]


def test_server_index_repo_mixes_languages(tmp_path):
    (tmp_path / "mod.py").write_text("import os\n\ndef f():\n    return os.getcwd()\n", encoding="utf-8")
    (tmp_path / "app.js").write_text(_read(JS_FIXTURE), encoding="utf-8")
    (tmp_path / "app.ts").write_text(_read(TS_FIXTURE), encoding="utf-8")

    idx = index_repo(str(tmp_path))
    assert idx["totals"]["files"] == 3
    assert "mod.py" in idx["files"]
    assert "Queue" in idx["files"]["app.js"]["classes"]
    assert "Worker" in idx["files"]["app.ts"]["classes"]
    # dependencies aggregate across languages
    assert {"os", "events", "http"} <= set(idx["dependencies"])


# ─── determinism + robustness ──────────────────────────────────────────────────

def test_extract_is_deterministic(js_source):
    a = ts.extract(js_source, "javascript")
    b = ts.extract(js_source, "javascript")
    assert a == b


def test_compress_is_deterministic(ts_source):
    assert ts.compress(ts_source, "typescript") == ts.compress(ts_source, "typescript")


def test_unsupported_language_raises():
    with pytest.raises(ValueError):
        ts.extract("const x = 1;", "ruby")


def test_broken_source_does_not_crash():
    # tree-sitter is error-tolerant: a malformed file still yields a result
    # rather than raising, so extract/compress degrade gracefully.
    broken = "function ( { const x =\nclass {"
    result = ts.extract(broken, "javascript")
    assert isinstance(result, dict)
    assert set(result) == {"imports", "classes", "fonctions"}
    compressed = ts.compress(broken, "javascript")
    assert isinstance(compressed, str)


def test_server_get_compressed_broken_js_returns_safely(tmp_path):
    f = tmp_path / "broken.js"
    f.write_text("function ( { const =", encoding="utf-8")
    result = get_compressed(str(f))
    # Either compresses what it can or returns an error dict — never raises.
    assert "compressed" in result or "error" in result
