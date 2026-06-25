"""
test_sse_transport.py — Tests pour le support SSE de refract-proxy.

Couvre :
  - Parsing CLI : --url, --transport, combinaisons invalides
  - _TargetClient._effective_transport() avec transport forcé
  - Retry + timeout SSE (mock réseau, pas de vrai serveur)
  - Parité compression : même résultat via "file" qu'avec transport="sse" simulé
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_cli import _build_parser, main
from refract_proxy import (
    RefractProxy,
    _TargetClient,
    _SSE_MAX_RETRIES,
    _SSE_RETRY_DELAY,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

FAKE_TOOLS = [
    {
        "name": "send_message",
        "description": "Send a message to a channel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Target channel"},
                "text": {"type": "string", "description": "Message body"},
            },
            "required": ["channel", "text"],
        },
    }
]


# ─── CLI : --url ──────────────────────────────────────────────────────────────

def test_parser_url_sets_url_attr():
    parser = _build_parser()
    args = parser.parse_args(["--url", "https://example.com/sse"])
    assert args.url == "https://example.com/sse"
    assert args.target is None


def test_parser_url_and_target_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--url", "https://x.com", "--target", "https://y.com"])


def test_parser_url_and_stdio_cmd_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--url", "https://x.com", "--stdio-cmd", "npx server"])


def test_parser_url_default_transport_is_none():
    """--url ne fixe pas args.transport — c'est _run() qui infère 'sse'."""
    parser = _build_parser()
    args = parser.parse_args(["--url", "https://example.com/sse"])
    assert args.transport is None


def test_parser_url_with_port_and_mode():
    parser = _build_parser()
    args = parser.parse_args([
        "--url", "https://example.com/sse",
        "--port", "9000",
        "--mode", "http",
    ])
    assert args.url == "https://example.com/sse"
    assert args.port == 9000
    assert args.mode == "http"


# ─── CLI : --transport ────────────────────────────────────────────────────────

def test_parser_transport_sse():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://example.com/sse", "--transport", "sse"])
    assert args.transport == "sse"


def test_parser_transport_stdio():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://x.com", "--transport", "stdio"])
    assert args.transport == "stdio"


def test_parser_transport_invalid_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "https://x.com", "--transport", "websocket"])


def test_parser_transport_defaults_to_none():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://x.com"])
    assert args.transport is None


# ─── CLI : --sse-timeout ──────────────────────────────────────────────────────

def test_parser_sse_timeout_default():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://x.com"])
    assert args.sse_timeout == 30.0


def test_parser_sse_timeout_custom():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://x.com", "--sse-timeout", "10"])
    assert args.sse_timeout == 10.0


# ─── CLI : validations ────────────────────────────────────────────────────────

def test_main_url_with_transport_stdio_errors():
    """--url + --transport stdio doit produire une erreur argparse."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--url", "https://example.com/sse", "--transport", "stdio"])
    assert exc_info.value.code != 0


def test_main_url_implies_sse_proxy_instantiation(monkeypatch, tmp_path):
    """main() avec --url doit créer un RefractProxy(transport='sse')."""
    received = {}

    import refract_proxy as rp

    original_init = rp.RefractProxy.__init__

    def capture_init(self, target_url, port=8080, verbose=False,
                     use_cache=False, transport=None, sse_timeout=30.0):
        received["transport"] = transport
        received["target_url"] = target_url
        # Appel minimal pour que connect() ne plante pas
        original_init(self, target_url, port=port, verbose=verbose,
                      use_cache=use_cache, transport=transport, sse_timeout=sse_timeout)

    async def mock_connect(self):
        self._tools = FAKE_TOOLS

    async def mock_serve(self):
        pass

    monkeypatch.setattr(rp.RefractProxy, "__init__", capture_init)
    monkeypatch.setattr(rp.RefractProxy, "connect", mock_connect)
    monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

    main(["--url", "https://example.com/sse"])

    assert received["transport"] == "sse"
    assert received["target_url"] == "https://example.com/sse"


def test_main_transport_sse_passed_to_proxy(monkeypatch, tmp_path):
    """--transport sse doit être transmis à RefractProxy."""
    import json

    schema_file = tmp_path / "t.json"
    schema_file.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")

    received = {}

    import refract_proxy as rp

    original_init = rp.RefractProxy.__init__

    def capture_init(self, target_url, port=8080, verbose=False,
                     use_cache=False, transport=None, sse_timeout=30.0):
        received["transport"] = transport
        original_init(self, target_url, port=port, verbose=verbose,
                      use_cache=use_cache, transport=transport, sse_timeout=sse_timeout)

    async def mock_connect(self):
        self._tools = FAKE_TOOLS

    async def mock_serve(self):
        pass

    monkeypatch.setattr(rp.RefractProxy, "__init__", capture_init)
    monkeypatch.setattr(rp.RefractProxy, "connect", mock_connect)
    monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

    main(["--target", str(schema_file), "--transport", "sse"])

    assert received["transport"] == "sse"


# ─── _TargetClient : _effective_transport() ──────────────────────────────────

def test_effective_transport_forced_sse_overrides_stdio_cmd():
    """Un URL qui ressemble à une commande stdio → SSE si forcé."""
    c = _TargetClient("npx @mcp/server /tmp", transport="sse")
    assert c._effective_transport() == "sse"


def test_effective_transport_forced_stdio_overrides_http():
    """Un URL HTTP → stdio si forcé (cas improbable mais l'API le permet)."""
    c = _TargetClient("https://example.com/sse", transport="stdio")
    assert c._effective_transport() == "stdio"


def test_effective_transport_auto_http_is_sse():
    c = _TargetClient("https://example.com/sse")
    assert c._effective_transport() == "sse"


def test_effective_transport_auto_npx_is_stdio():
    c = _TargetClient("npx @mcp/server /tmp")
    assert c._effective_transport() == "stdio"


def test_effective_transport_auto_file(tmp_path):
    f = tmp_path / "t.json"
    f.write_text("[]")
    c = _TargetClient(str(f))
    assert c._effective_transport() == "file"


def test_effective_transport_forced_sse_http_url():
    """Transport forcé sur un URL HTTP → toujours sse (redondant mais correct)."""
    c = _TargetClient("https://api.example.com/sse", transport="sse")
    assert c._effective_transport() == "sse"


# ─── _TargetClient : dispatch via transport forcé ────────────────────────────

def test_list_tools_uses_forced_sse(monkeypatch):
    """Avec transport='sse', list_tools() doit appeler _sse_list_tools()."""
    called = []

    async def mock_sse_list(self):
        called.append("sse")
        return FAKE_TOOLS

    monkeypatch.setattr(_TargetClient, "_sse_list_tools", mock_sse_list)

    c = _TargetClient("npx @mcp/server", transport="sse")

    async def _run():
        return await c.list_tools()

    result = asyncio.run(_run())
    assert "sse" in called
    assert result == FAKE_TOOLS


def test_call_tool_uses_forced_sse(monkeypatch):
    """Avec transport='sse', call_tool() doit appeler _sse_call_tool()."""
    called = []

    async def mock_sse_call(self, name, arguments):
        called.append(("sse", name))
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(_TargetClient, "_sse_call_tool", mock_sse_call)

    c = _TargetClient("npx @mcp/server", transport="sse")

    async def _run():
        return await c.call_tool("send_message", {"channel": "#x", "text": "hi"})

    result = asyncio.run(_run())
    assert ("sse", "send_message") in called
    assert result[0]["text"] == "ok"


# ─── SSE retry + timeout ─────────────────────────────────────────────────────

def test_sse_list_tools_retries_on_error(monkeypatch):
    """_sse_list_tools() doit réessayer _SSE_MAX_RETRIES fois avant de lever."""
    attempts = []

    async def fail_sse(*args, **kwargs):
        raise OSError("connection refused")

    async def no_sleep(_):
        pass

    # Patcher sse_client dans refract_proxy
    import refract_proxy as rp
    monkeypatch.setattr(rp, "_SSE_RETRY_DELAY", 0.0)

    import unittest.mock as mock

    # On patche le module mcp.client.sse.sse_client utilisé dans _sse_list_tools
    with mock.patch("mcp.client.sse.sse_client", side_effect=OSError("refused")):
        c = _TargetClient("https://example.com/sse", transport="sse", timeout=1.0)

        async def _run():
            with pytest.raises(RuntimeError, match="failed after"):
                await c._sse_list_tools()

        asyncio.run(_run())


def test_sse_list_tools_succeeds_on_third_attempt(monkeypatch):
    """_sse_list_tools() doit réussir si la 3e tentative fonctionne."""
    import unittest.mock as mock
    from contextlib import asynccontextmanager
    from anyio.streams.memory import (
        MemoryObjectReceiveStream,
        MemoryObjectSendStream,
    )

    attempt_count = [0]

    # Mock de ClientSession
    class _MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            class _Result:
                tools = []
            return _Result()

    @asynccontextmanager
    async def flaky_sse_client(url, **kwargs):
        attempt_count[0] += 1
        if attempt_count[0] < _SSE_MAX_RETRIES:
            raise OSError("transient error")
        # Success on last attempt — yield fake (read, write) streams
        yield (object(), object())

    import refract_proxy as rp
    monkeypatch.setattr(rp, "_SSE_RETRY_DELAY", 0.0)

    with (
        mock.patch("mcp.client.sse.sse_client", flaky_sse_client),
        mock.patch("mcp.client.session.ClientSession", lambda r, w: _MockSession()),
    ):
        c = _TargetClient("https://example.com/sse", transport="sse", timeout=1.0)

        async def _run():
            return await c._sse_list_tools()

        result = asyncio.run(_run())

    assert attempt_count[0] == _SSE_MAX_RETRIES
    assert result == []


def test_sse_call_tool_raises_after_all_retries(monkeypatch):
    """_sse_call_tool() doit lever RuntimeError après _SSE_MAX_RETRIES échecs."""
    import unittest.mock as mock

    import refract_proxy as rp
    monkeypatch.setattr(rp, "_SSE_RETRY_DELAY", 0.0)

    with mock.patch("mcp.client.sse.sse_client", side_effect=ConnectionError("refused")):
        c = _TargetClient("https://example.com/sse", transport="sse", timeout=1.0)

        async def _run():
            with pytest.raises(RuntimeError, match="failed after"):
                await c._sse_call_tool("my_tool", {})

        asyncio.run(_run())


# ─── compression identique quel que soit le transport ─────────────────────────

def test_compression_identical_regardless_of_transport(tmp_path):
    """La compression doit produire le même résultat que le transport soit
    'file' (auto-détecté) ou 'sse' (forcé sur un fichier simulé).

    On simule le transport SSE en patchant _sse_list_tools pour qu'il lise
    le même fichier JSON que le transport 'file'.
    """
    import asyncio
    from unittest import mock
    from refract_proxy import RefractProxy, _compressed_to_input_schema
    from mcp_optimizer import compress_tool, collect_defs, build_index

    schema_file = tmp_path / "tools.json"
    schema_file.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")

    # Proxy "file" (auto-détecté)
    proxy_file = RefractProxy(target_url=str(schema_file))
    asyncio.run(proxy_file.connect())

    # Proxy "SSE" — on simule list_tools qui retourne les mêmes tools
    proxy_sse = RefractProxy(target_url="https://fake.example.com/sse", transport="sse")

    # patch.object sur une instance ne lie pas self — la signature ne prend pas self
    async def mock_sse_list_tools():
        return FAKE_TOOLS

    with mock.patch.object(proxy_sse._client, "_sse_list_tools", mock_sse_list_tools):
        asyncio.run(proxy_sse.connect())

    # Les deux proxies doivent produire les mêmes schemas compressés
    for tool in FAKE_TOOLS:
        name = tool["name"]
        schema_via_file = _compressed_to_input_schema(proxy_file._compressed[name])
        schema_via_sse = _compressed_to_input_schema(proxy_sse._compressed[name])
        assert schema_via_file == schema_via_sse, (
            f"Compression diverge pour {name!r} entre transport 'file' et 'sse'"
        )


# ─── RefractProxy : transport + sse_timeout passés à _TargetClient ───────────

def test_proxy_passes_transport_to_client():
    proxy = RefractProxy("https://example.com/sse", transport="sse")
    assert proxy._client._forced_transport == "sse"


def test_proxy_passes_sse_timeout_to_client():
    proxy = RefractProxy("https://example.com/sse", transport="sse", sse_timeout=15.0)
    assert proxy._client._timeout == 15.0


def test_proxy_cmd_parts_none_when_transport_sse():
    """Transport forcé à SSE → _cmd_parts doit être None même si l'URL ressemble à une cmd."""
    proxy = RefractProxy("npx @mcp/server /tmp", transport="sse")
    assert proxy._cmd_parts is None


def test_proxy_cmd_parts_set_when_transport_stdio():
    proxy = RefractProxy("npx @mcp/server /tmp", transport="stdio")
    assert proxy._cmd_parts == ["npx", "@mcp/server", "/tmp"]
