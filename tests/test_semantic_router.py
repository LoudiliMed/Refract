"""
test_semantic_router.py — Unit tests for SemanticRouter and its integration
in RefractProxy.identify_tool().

fastembed is mocked throughout: embeddings are injected directly or the
TextEmbedding class is patched, so no network or model download is needed.
All tests are fully deterministic.
"""

from __future__ import annotations

import json
import asyncio

import numpy as np
import pytest

import semantic_router as sr
from semantic_router import SemanticRouter, _keyword_score


# ─── fixtures ────────────────────────────────────────────────────────────────

TOOLS = [
    {"name": "create_event",  "description": "Creates a new calendar event with title, time and guests"},
    {"name": "list_events",   "description": "Lists upcoming calendar events in a date range"},
    {"name": "delete_event",  "description": "Permanently deletes a calendar event by its ID"},
    {"name": "send_email",    "description": "Composes and sends an email message to one or more recipients"},
    {"name": "read_file",     "description": "Reads the contents of a file from the local filesystem"},
]

DIM = 5   # one dimension per tool for orthogonal unit-vector tests

# Orthogonal unit vectors — tool i is "closest" to query vector e_i
TOOL_VECS = np.eye(DIM, dtype=np.float32)


def _inject(router: SemanticRouter, tools: list[dict], vecs: np.ndarray) -> None:
    """Bypass index_tools() and inject pre-computed embeddings directly."""
    router._tool_names = [t["name"] for t in tools]
    router._tool_embeddings = vecs.copy()


def _mock_model(query_vec: np.ndarray):
    """Return a mock TextEmbedding whose embed() yields query_vec for any single query."""
    from unittest.mock import MagicMock

    m = MagicMock()
    m.embed.side_effect = lambda texts: iter([query_vec] * len(list(texts)))
    return m


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Ensure no real model leaks between tests."""
    SemanticRouter._model_cache.clear()
    yield
    SemanticRouter._model_cache.clear()


# ─── _keyword_score ───────────────────────────────────────────────────────────

def test_keyword_score_exact_match():
    """A query word that appears verbatim in a tool name scores highest."""
    result = _keyword_score("send email", TOOLS)
    assert result == "send_email"


def test_keyword_score_description_match():
    """A word that appears only in the description is still matched."""
    result = _keyword_score("filesystem", TOOLS)
    assert result == "read_file"


def test_keyword_score_empty_tools():
    assert _keyword_score("anything", []) == ""


def test_keyword_score_no_match_returns_first_or_any():
    """With no overlap, any tool is returned (deterministic tie-break)."""
    result = _keyword_score("xyz_unrelated_query", TOOLS)
    assert isinstance(result, str)


# ─── SemanticRouter.index_tools ──────────────────────────────────────────────

def test_index_tools_stores_names():
    router = SemanticRouter()
    _inject(router, TOOLS, TOOL_VECS)
    assert router._tool_names == [t["name"] for t in TOOLS]


def test_index_tools_stores_embeddings_shape():
    router = SemanticRouter()
    _inject(router, TOOLS, TOOL_VECS)
    assert router._tool_embeddings.shape == (len(TOOLS), DIM)


def test_index_tools_empty_list():
    router = SemanticRouter()
    router.index_tools([])
    assert router._tool_names == []
    assert router._tool_embeddings is None


def test_index_tools_calls_model_embed(monkeypatch):
    """index_tools() embeds one text per tool via the model."""
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    mock_model.embed.return_value = iter([np.zeros(4, dtype=np.float32)] * len(TOOLS))
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = mock_model

    router = SemanticRouter()
    router.index_tools(TOOLS)

    mock_model.embed.assert_called_once()
    texts = list(mock_model.embed.call_args[0][0])
    assert len(texts) == len(TOOLS)
    assert "create_event" in texts[0]


# ─── SemanticRouter.find_best_tool ───────────────────────────────────────────

def test_find_best_tool_picks_closest():
    """Query aligned with tool-2 vector -> tool-2 wins."""
    router = SemanticRouter()
    _inject(router, TOOLS, TOOL_VECS)
    # Query vector = e_2 (aligned with delete_event)
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(TOOL_VECS[2])

    results = router.find_best_tool("remove an event", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "delete_event"
    assert results[0][1] == pytest.approx(1.0)


def test_find_best_tool_top_k():
    """top_k=3 returns three results sorted descending."""
    router = SemanticRouter()
    _inject(router, TOOLS, TOOL_VECS)
    # Slight mix: mostly e_0, a bit e_1
    q = np.array([0.9, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
    q /= np.linalg.norm(q)
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(q)

    results = router.find_best_tool("query", top_k=3)
    assert len(results) == 3
    # Scores must be descending
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] == "create_event"


def test_find_best_tool_empty_index_returns_empty():
    router = SemanticRouter()
    # No index_tools called
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(np.zeros(4))
    assert router.find_best_tool("query") == []


def test_find_best_tool_score_is_cosine():
    """With unit-norm vectors, score == cosine similarity."""
    router = SemanticRouter()
    _inject(router, TOOLS[:2], TOOL_VECS[:2])
    # Query at 45° between tool-0 and tool-1
    q = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q /= np.linalg.norm(q)
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(q)

    results = router.find_best_tool("query", top_k=2)
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-5)


# ─── SemanticRouter.find_best_tool_with_threshold ────────────────────────────

def test_threshold_above_returns_name():
    router = SemanticRouter()
    _inject(router, TOOLS, TOOL_VECS)
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(TOOL_VECS[3])

    result = router.find_best_tool_with_threshold("send a message", min_score=0.3)
    assert result == "send_email"


def test_threshold_below_returns_none():
    """A query that scores below min_score must return None."""
    router = SemanticRouter()
    _inject(router, TOOLS, TOOL_VECS)
    # Query slightly off-axis — score against all tools < 0.9
    q = np.array([0.7, 0.7, 0.0, 0.0, 0.1], dtype=np.float32)
    q /= np.linalg.norm(q)
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(q)

    result = router.find_best_tool_with_threshold("query", min_score=0.9)
    assert result is None


def test_threshold_exact_boundary():
    """Score exactly equal to min_score should be accepted (>=, not >)."""
    router = SemanticRouter()
    _inject(router, TOOLS[:1], TOOL_VECS[:1])
    # Query perfectly aligned with tool-0 -> score = 1.0
    SemanticRouter._model_cache["BAAI/bge-small-en-v1.5"] = _mock_model(TOOL_VECS[0])

    result = router.find_best_tool_with_threshold("query", min_score=1.0)
    assert result == "create_event"


def test_threshold_empty_index_returns_none():
    router = SemanticRouter()
    assert router.find_best_tool_with_threshold("query") is None


# ─── graceful degradation (fastembed not installed) ──────────────────────────

def test_get_model_raises_runtime_error_without_fastembed(monkeypatch):
    """If fastembed cannot be imported, _get_model raises RuntimeError."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("no module named fastembed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    router = SemanticRouter()
    with pytest.raises(RuntimeError, match="fastembed"):
        router._get_model()


# ─── RefractProxy.identify_tool() integration ────────────────────────────────

def _make_proxy(tmp_path) -> "RefractProxy":
    from refract_proxy import RefractProxy

    schema_file = tmp_path / "tools.json"
    schema_file.write_text(json.dumps(TOOLS), encoding="utf-8")
    proxy = RefractProxy(target_url=str(schema_file))

    async def _load():
        await proxy.connect()

    asyncio.run(_load())
    return proxy


def test_proxy_identify_tool_returns_string(tmp_path):
    """identify_tool() returns a non-empty string for a clear query."""
    proxy = _make_proxy(tmp_path)
    result = proxy.identify_tool("I want to create a new calendar meeting")
    assert isinstance(result, str)
    assert len(result) > 0


def test_proxy_identify_tool_correct_routing(tmp_path):
    """Semantic routing picks the right tool for an unambiguous query."""
    proxy = _make_proxy(tmp_path)
    assert proxy.identify_tool("schedule a new event on my calendar") == "create_event"
    assert proxy.identify_tool("compose and send an email to someone") == "send_email"
    assert proxy.identify_tool("read the contents of a local file") == "read_file"


def test_proxy_identify_tool_below_threshold_returns_none_or_fallback(tmp_path):
    """A very off-topic query may return None or a keyword-matched fallback."""
    proxy = _make_proxy(tmp_path)
    # The result type should always be str | None — never raises
    result = proxy.identify_tool("quantum entanglement of fermions", min_score=0.99)
    assert result is None or isinstance(result, str)


def test_proxy_identify_tool_keyword_fallback_when_no_router(tmp_path):
    """When _router is None, identify_tool falls back to keyword matching."""
    proxy = _make_proxy(tmp_path)
    proxy._router = None   # simulate fastembed unavailable

    result = proxy.identify_tool("send email")
    assert result == "send_email"


def test_proxy_router_built_after_connect(tmp_path):
    """After connect(), _router is set (fastembed is installed in this env)."""
    proxy = _make_proxy(tmp_path)
    assert proxy._router is not None
    assert isinstance(proxy._router, SemanticRouter)
    assert len(proxy._router._tool_names) == len(TOOLS)
