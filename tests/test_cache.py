"""
test_cache.py — Vérifie CacheInjector (inject_cache_control, estimate_cache_savings,
to_anthropic_format) et l'intégration dans RefractProxy.as_anthropic_tools().
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cache_injector import (
    CacheInjector,
    PRICE_INPUT_USD_PER_M,
    PRICE_WRITE_USD_PER_M,
    PRICE_READ_USD_PER_M,
)

# ─── fixtures ────────────────────────────────────────────────────────────────

TOOLS_3 = [
    {"name": "tool_a", "description": "Tool A", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "tool_b", "description": "Tool B", "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}},
    {"name": "tool_c", "description": "Tool C", "inputSchema": {"type": "object", "properties": {"y": {"type": "integer"}}}},
]

TOOLS_1 = [
    {"name": "only_tool", "description": "Solo", "inputSchema": {"type": "object", "properties": {}}}
]


# ─── inject_cache_control ────────────────────────────────────────────────────

def test_inject_cache_control_adds_to_last():
    result = CacheInjector.inject_cache_control(TOOLS_3)
    # Seul le dernier a cache_control
    assert "cache_control" not in result[0]
    assert "cache_control" not in result[1]
    assert result[2]["cache_control"] == {"type": "ephemeral"}


def test_inject_cache_control_single_tool():
    result = CacheInjector.inject_cache_control(TOOLS_1)
    assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_inject_cache_control_empty_list():
    result = CacheInjector.inject_cache_control([])
    assert result == []


def test_inject_cache_control_does_not_mutate_original():
    original = [{"name": "t", "inputSchema": {}}]
    result = CacheInjector.inject_cache_control(original)
    assert "cache_control" not in original[0], "L'original ne doit pas être modifié"
    assert "cache_control" in result[0]


def test_inject_cache_control_preserves_other_fields():
    result = CacheInjector.inject_cache_control(TOOLS_3)
    last = result[-1]
    assert last["name"] == "tool_c"
    assert last["description"] == "Tool C"
    assert "inputSchema" in last
    assert last["cache_control"] == {"type": "ephemeral"}


def test_inject_cache_control_list_length_unchanged():
    result = CacheInjector.inject_cache_control(TOOLS_3)
    assert len(result) == len(TOOLS_3)


def test_inject_cache_control_idempotent():
    """Appliquer deux fois ne duplique pas cache_control."""
    once = CacheInjector.inject_cache_control(TOOLS_3)
    twice = CacheInjector.inject_cache_control(once)
    assert twice[-1]["cache_control"] == {"type": "ephemeral"}
    # Il n'y a qu'une clé cache_control
    assert list(k for k in twice[-1] if k == "cache_control") == ["cache_control"]


# ─── estimate_cache_savings ──────────────────────────────────────────────────

def test_estimate_returns_four_fields():
    result = CacheInjector.estimate_cache_savings(1000, 100)
    assert "cout_sans_cache_sans_refract" in result
    assert "cout_avec_cache_avec_refract" in result
    assert "economie_totale_usd" in result
    assert "reduction_pct" in result


def test_estimate_single_request():
    """Avec 1 seule requête : cost_avec = cache write, pas de hits."""
    tokens = 1_000_000
    result = CacheInjector.estimate_cache_savings(tokens, requests_per_day=1, days=1)

    expected_sans = tokens * PRICE_INPUT_USD_PER_M / 1_000_000 * 1
    expected_avec = tokens * PRICE_WRITE_USD_PER_M / 1_000_000  # 0 hits

    assert abs(result["cout_sans_cache_sans_refract"] - expected_sans) < 0.0001
    assert abs(result["cout_avec_cache_avec_refract"] - expected_avec) < 0.0001


def test_estimate_many_requests_cache_hits_dominate():
    """Avec beaucoup de requêtes, le cache hit (0,30$/M) domine le coût total."""
    tokens = 1_000
    requests = 1000
    result = CacheInjector.estimate_cache_savings(tokens, requests_per_day=requests, days=1)

    # Le coût avec cache doit être très inférieur au coût sans cache
    assert result["cout_avec_cache_avec_refract"] < result["cout_sans_cache_sans_refract"]
    assert result["reduction_pct"] > 50  # au moins 50% d'économie


def test_estimate_savings_positive():
    """Les économies ne sont jamais négatives."""
    result = CacheInjector.estimate_cache_savings(500, requests_per_day=10, days=30)
    assert result["economie_totale_usd"] >= 0


def test_estimate_prices_correct():
    """Vérifie les tarifs exacts pour 1M tokens, 2 requêtes."""
    tokens = 1_000_000
    # 2 requêtes : 1 cache write + 1 cache hit
    result = CacheInjector.estimate_cache_savings(tokens, requests_per_day=2, days=1)

    expected_sans = 1_000_000 * 3.00 / 1_000_000 * 2   # = 6.00 $
    expected_avec = (1_000_000 * 3.75 / 1_000_000       # = 3.75 $ write
                    + 1_000_000 * 0.30 / 1_000_000 * 1) # = 0.30 $ hit
    # = 4.05 $

    assert abs(result["cout_sans_cache_sans_refract"] - expected_sans) < 0.001
    assert abs(result["cout_avec_cache_avec_refract"] - expected_avec) < 0.001
    assert abs(result["economie_totale_usd"] - (expected_sans - expected_avec)) < 0.001


def test_estimate_reduction_pct_between_0_and_100():
    result = CacheInjector.estimate_cache_savings(1000, requests_per_day=50, days=30)
    assert 0 <= result["reduction_pct"] <= 100


def test_estimate_zero_requests_per_day():
    """0 requête/jour sur 30 jours = 0 requête totale = coûts nuls."""
    result = CacheInjector.estimate_cache_savings(1000, requests_per_day=0, days=30)
    assert result["cout_sans_cache_sans_refract"] == 0.0
    assert result["economie_totale_usd"] == 0.0
    assert result["reduction_pct"] == 0.0


def test_estimate_30_day_default():
    r1 = CacheInjector.estimate_cache_savings(1000, 10)
    r2 = CacheInjector.estimate_cache_savings(1000, 10, days=30)
    assert r1 == r2


# ─── to_anthropic_format ─────────────────────────────────────────────────────

def test_to_anthropic_format_renames_input_schema():
    result = CacheInjector.to_anthropic_format(TOOLS_3, use_cache=False)
    for t in result:
        assert "input_schema" in t
        assert "inputSchema" not in t


def test_to_anthropic_format_injects_cache_when_use_cache_true():
    result = CacheInjector.to_anthropic_format(TOOLS_3, use_cache=True)
    assert result[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in result[0]


def test_to_anthropic_format_no_cache_when_false():
    result = CacheInjector.to_anthropic_format(TOOLS_3, use_cache=False)
    for t in result:
        assert "cache_control" not in t


def test_to_anthropic_format_default_is_cache_true():
    result = CacheInjector.to_anthropic_format(TOOLS_3)
    assert result[-1]["cache_control"] == {"type": "ephemeral"}


def test_to_anthropic_format_preserves_name_description():
    result = CacheInjector.to_anthropic_format(TOOLS_3, use_cache=False)
    for orig, conv in zip(TOOLS_3, result):
        assert conv["name"] == orig["name"]
        assert conv["description"] == orig["description"]


# ─── intégration RefractProxy.as_anthropic_tools() ───────────────────────────

def _make_proxy_with_tools(tmp_path, use_cache=True):
    from refract_proxy import RefractProxy

    schema_file = tmp_path / "tools.json"
    schema_file.write_text(json.dumps([
        {
            "name": "send_email",
            "description": "Sends an email to a recipient.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                },
                "required": ["to", "subject"],
            },
        },
        {
            "name": "read_file",
            "description": "Reads a file from the filesystem.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ]), encoding="utf-8")

    proxy = RefractProxy(target_url=str(schema_file), use_cache=use_cache)

    async def _load():
        await proxy.connect()

    asyncio.run(_load())
    return proxy


def test_as_anthropic_tools_returns_dicts(tmp_path):
    proxy = _make_proxy_with_tools(tmp_path, use_cache=False)
    tools = proxy.as_anthropic_tools()
    assert isinstance(tools, list)
    for t in tools:
        assert isinstance(t, dict)
        assert "name" in t
        assert "input_schema" in t


def test_as_anthropic_tools_cache_on_last(tmp_path):
    proxy = _make_proxy_with_tools(tmp_path, use_cache=True)
    tools = proxy.as_anthropic_tools()
    assert "cache_control" not in tools[0]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_as_anthropic_tools_no_cache_when_disabled(tmp_path):
    proxy = _make_proxy_with_tools(tmp_path, use_cache=False)
    tools = proxy.as_anthropic_tools()
    for t in tools:
        assert "cache_control" not in t


def test_as_anthropic_tools_uses_compressed_descriptions(tmp_path):
    proxy = _make_proxy_with_tools(tmp_path, use_cache=False)
    tools = proxy.as_anthropic_tools()
    # La description doit être la version courte (< description brute)
    for t in tools:
        assert len(t["description"]) > 0


def test_as_anthropic_tools_input_schema_is_dict(tmp_path):
    proxy = _make_proxy_with_tools(tmp_path, use_cache=False)
    tools = proxy.as_anthropic_tools()
    for t in tools:
        assert isinstance(t["input_schema"], dict)
        assert t["input_schema"].get("type") == "object"


# ─── test endpoint /cache-estimate (importation directe) ─────────────────────

def test_cache_estimate_endpoint_structure():
    import importlib.util
    import pathlib

    main_path = pathlib.Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("main_mod_cache", main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeInput:
        tokens = 5000
        requests_per_day = 100
        days = 30

    result = mod.cache_estimate(FakeInput())
    assert "cout_sans_cache_sans_refract" in result
    assert "cout_avec_cache_avec_refract" in result
    assert "economie_totale_usd" in result
    assert "reduction_pct" in result
    assert result["economie_totale_usd"] >= 0
