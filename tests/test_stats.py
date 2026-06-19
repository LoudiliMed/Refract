"""
test_stats.py — Vérifie que les stats s'accumulent correctement
dans le proxy (refract_proxy._PROXY_STATS) et dans main.py (_SESSION_STATS).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(__file__), "..")))


# ─── tests stats du proxy ────────────────────────────────────────────────────

def test_proxy_stats_start_empty():
    import refract_proxy as rp
    rp._PROXY_STATS.clear()
    stats = rp.get_proxy_stats()
    assert stats["total_requests"] == 0
    assert stats["tokens_economises"] == 0
    assert stats["cout_evite_usd"] == 0.0
    assert stats["reduction_moyenne_pct"] == 0.0
    assert stats["par_serveur"] == {}


def test_proxy_stats_accumulate():
    import refract_proxy as rp
    rp._PROXY_STATS.clear()

    rp._update_stats("https://server-a.com", tokens_raw=1000, tokens_compressed=100)
    stats = rp.get_proxy_stats()

    assert stats["total_requests"] == 1
    assert stats["tokens_economises"] == 900
    assert stats["reduction_moyenne_pct"] == 90.0
    assert "https://server-a.com" in stats["par_serveur"]
    assert stats["par_serveur"]["https://server-a.com"]["requests"] == 1
    assert stats["par_serveur"]["https://server-a.com"]["tokens_economises"] == 900


def test_proxy_stats_multiple_servers():
    import refract_proxy as rp
    rp._PROXY_STATS.clear()

    rp._update_stats("https://server-a.com", 1000, 100)
    rp._update_stats("https://server-b.com", 2000, 200)
    rp._update_stats("https://server-a.com", 500, 50)

    stats = rp.get_proxy_stats()
    assert stats["total_requests"] == 3
    assert stats["tokens_economises"] == 900 + 1800 + 450
    assert len(stats["par_serveur"]) == 2
    assert stats["par_serveur"]["https://server-a.com"]["requests"] == 2
    assert stats["par_serveur"]["https://server-b.com"]["requests"] == 1


def test_proxy_stats_no_negative_savings():
    """Les tokens économisés ne doivent jamais être négatifs."""
    import refract_proxy as rp
    rp._PROXY_STATS.clear()

    # Cas où compressé > brut (ne devrait pas arriver mais on l'ignore)
    rp._update_stats("https://server.com", tokens_raw=100, tokens_compressed=200)
    stats = rp.get_proxy_stats()
    assert stats["tokens_economises"] == 0, "Pas d'économies négatives"


def test_proxy_stats_cout_evite_usd():
    """Le coût évité est calculé à $3/million tokens."""
    import refract_proxy as rp
    rp._PROXY_STATS.clear()

    rp._update_stats("https://server.com", tokens_raw=1_000_000, tokens_compressed=0)
    stats = rp.get_proxy_stats()
    assert abs(stats["cout_evite_usd"] - 3.0) < 0.001


# ─── tests stats du proxy via tools/list ────────────────────────────────────

def test_proxy_stats_updated_by_tools_list(tmp_path):
    """handle_tools_list() doit mettre à jour _PROXY_STATS."""
    import asyncio
    import refract_proxy as rp

    rp._PROXY_STATS.clear()

    schema_file = tmp_path / "tools.json"
    schema_file.write_text(json.dumps([
        {
            "name": "send_email",
            "description": "Sends an email message to the specified recipient with subject and body.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body content"},
                },
                "required": ["to", "subject", "body"],
            },
        }
    ]), encoding="utf-8")

    proxy = rp.RefractProxy(target_url=str(schema_file))

    async def _run():
        await proxy.connect()

    asyncio.run(_run())
    proxy.handle_tools_list()

    stats = rp.get_proxy_stats()
    assert stats["total_requests"] >= 1
    assert str(schema_file) in stats["par_serveur"]


# ─── tests endpoint GET /stats de main.py ───────────────────────────────────

def test_stats_endpoint_structure():
    """GET /stats retourne la structure attendue."""
    # Import sans démarrer uvicorn
    import importlib
    import importlib.util
    import pathlib

    main_path = pathlib.Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("main_module", main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Réinitialiser les stats
    mod._SESSION_STATS["total_requests"] = 0
    mod._SESSION_STATS["tokens_economises"] = 0
    mod._SESSION_STATS["_tokens_raw"] = 0
    mod._SESSION_STATS["par_serveur"] = {}

    result = mod.get_stats()
    assert "session_stats" in result
    ss = result["session_stats"]
    assert "total_requests" in ss
    assert "tokens_economises" in ss
    assert "cout_evite_usd" in ss
    assert "reduction_moyenne_pct" in ss
    assert "par_serveur" in ss


def test_stats_endpoint_accumulates():
    """Les stats s'accumulent après des appels à _record_stats."""
    import importlib
    import importlib.util
    import pathlib

    main_path = pathlib.Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("main_module2", main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod._SESSION_STATS["total_requests"] = 0
    mod._SESSION_STATS["tokens_economises"] = 0
    mod._SESSION_STATS["_tokens_raw"] = 0
    mod._SESSION_STATS["par_serveur"] = {}

    mod._record_stats("calendar.json", tokens_raw=5000, tokens_compressed=300)
    mod._record_stats("calendar.json", tokens_raw=5000, tokens_compressed=300)

    result = mod.get_stats()
    ss = result["session_stats"]
    assert ss["total_requests"] == 2
    assert ss["tokens_economises"] == (5000 - 300) * 2
    assert ss["reduction_moyenne_pct"] > 0
    assert "calendar.json" in ss["par_serveur"]
    assert ss["par_serveur"]["calendar.json"]["requests"] == 2
