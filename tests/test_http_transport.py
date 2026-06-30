"""
test_http_transport.py — Tests for the Streamable HTTP transport (MCP spec 2025-03-26).

Covers:
  - CLI: all 3 transports accepted, --transport http requires an HTTP URL
  - _TargetClient: explicit transport="http" dispatches to _http_list_tools/_http_call_tool
  - Compression is transport-agnostic (same input → same output on all 3 paths)
  - Mock of streamablehttp_client (no real server required)
  - ExceptionGroup unwrapping helper
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_cli import _build_parser, _validate_transport, main
from refract_proxy import (
    RefractProxy,
    _TargetClient,
    _unwrap_exception_group,
    _compressed_to_input_schema,
)
from mcp_optimizer import compress_tool, collect_defs

# ─── shared fixture tools ────────────────────────────────────────────────────

FAKE_TOOLS = [
    {
        "name": "send_message",
        "description": "Send a message to a channel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Target channel"},
                "text": {"type": "string", "description": "Message text"},
                "urgent": {"type": "boolean", "description": "Flag as urgent"},
            },
            "required": ["channel", "text"],
        },
    },
    {
        "name": "list_channels",
        "description": "List available channels.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results"},
            },
        },
    },
]


# ─── CLI: transport flag parsing ─────────────────────────────────────────────

class TestCLITransportParsing:
    def _parser(self):
        return _build_parser()

    def test_transport_default_is_none(self):
        args = self._parser().parse_args(["--target", "https://x.com"])
        assert args.transport is None

    def test_transport_stdio_accepted(self):
        args = self._parser().parse_args(
            ["--target", "https://x.com", "--transport", "stdio"]
        )
        assert args.transport == "stdio"

    def test_transport_sse_accepted(self):
        args = self._parser().parse_args(
            ["--target", "https://x.com", "--transport", "sse"]
        )
        assert args.transport == "sse"

    def test_transport_http_accepted(self):
        args = self._parser().parse_args(
            ["--target", "https://x.com/mcp", "--transport", "http"]
        )
        assert args.transport == "http"

    def test_transport_invalid_value_rejected(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--target", "https://x.com", "--transport", "websocket"])

    def test_transport_http_requires_http_url(self):
        parser = self._parser()
        args = parser.parse_args(["--target", "not-a-url", "--transport", "http"])
        with pytest.raises(SystemExit):
            _validate_transport(parser, args)

    def test_transport_sse_requires_http_url(self):
        parser = self._parser()
        args = parser.parse_args(["--stdio-cmd", "npx mcp-server", "--transport", "sse"])
        with pytest.raises(SystemExit):
            _validate_transport(parser, args)

    def test_transport_http_with_valid_url_passes_validation(self):
        parser = self._parser()
        args = parser.parse_args(["--target", "https://my-server.com/mcp", "--transport", "http"])
        _validate_transport(parser, args)  # must not raise

    def test_transport_stdio_with_command_passes_validation(self):
        parser = self._parser()
        args = parser.parse_args(["--stdio-cmd", "npx mcp-server", "--transport", "stdio"])
        _validate_transport(parser, args)  # must not raise

    def test_transport_none_with_command_passes_validation(self):
        parser = self._parser()
        args = parser.parse_args(["--stdio-cmd", "npx mcp-server"])
        _validate_transport(parser, args)  # must not raise


# ─── _TargetClient: transport dispatch ──────────────────────────────────────

class TestTargetClientTransportDispatch:
    def test_transport_none_http_url_falls_back_to_sse(self):
        """Without explicit transport, HTTP URL auto-detects as SSE (backward compat)."""
        client = _TargetClient("https://server.com/sse")
        assert client.transport is None
        assert not client._is_local_file()
        assert not client._is_stdio_cmd()

    def test_explicit_http_transport_stored(self):
        client = _TargetClient("https://server.com/mcp", transport="http")
        assert client.transport == "http"

    def test_explicit_sse_transport_stored(self):
        client = _TargetClient("https://server.com/sse", transport="sse")
        assert client.transport == "sse"

    def test_explicit_stdio_transport_stored(self):
        client = _TargetClient("npx mcp-server", transport="stdio")
        assert client.transport == "stdio"

    @pytest.mark.asyncio
    async def test_list_tools_http_transport_calls_http_method(self):
        client = _TargetClient("https://server.com/mcp", transport="http")
        called = []

        async def mock_http_list():
            called.append("http_list")
            return []

        client._http_list_tools = mock_http_list
        await client.list_tools()
        assert called == ["http_list"]

    @pytest.mark.asyncio
    async def test_call_tool_http_transport_calls_http_method(self):
        client = _TargetClient("https://server.com/mcp", transport="http")
        called = []

        async def mock_http_call(name, args):
            called.append(("http_call", name))
            return []

        client._http_call_tool = mock_http_call
        await client.call_tool("my_tool", {})
        assert called == [("http_call", "my_tool")]

    @pytest.mark.asyncio
    async def test_list_tools_sse_transport_calls_sse_method(self):
        client = _TargetClient("https://server.com/sse", transport="sse")
        called = []

        async def mock_sse_list():
            called.append("sse_list")
            return []

        client._sse_list_tools = mock_sse_list
        await client.list_tools()
        assert called == ["sse_list"]

    @pytest.mark.asyncio
    async def test_list_tools_local_file_bypasses_transport(self, tmp_path):
        """Local JSON file is always handled regardless of transport param."""
        f = tmp_path / "tools.json"
        f.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")
        client = _TargetClient(str(f), transport="http")
        result = await client.list_tools()
        assert len(result) == 2


# ─── Mock of streamablehttp_client ───────────────────────────────────────────

def _make_mock_session(tools: list[dict]):
    """Builds a mock ClientSession that returns the given tools list."""
    tool_objects = []
    for t in tools:
        obj = MagicMock()
        obj.name = t["name"]
        obj.description = t.get("description", "")
        obj.inputSchema = t.get("inputSchema", {})
        tool_objects.append(obj)

    list_result = MagicMock()
    list_result.tools = tool_objects

    call_result = MagicMock()
    call_result.content = [MagicMock(type="text", text="ok")]

    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=list_result)
    session.call_tool = AsyncMock(return_value=call_result)
    return session


@asynccontextmanager
async def _mock_streamablehttp_client(url):
    """Async context manager that mimics streamablehttp_client(url) -> (read, write, get_id)."""
    yield (MagicMock(), MagicMock(), lambda: "session-id")


class TestHttpTransportMocked:
    @pytest.mark.asyncio
    async def test_http_list_tools_uses_streamablehttp_client(self):
        client = _TargetClient("https://server.com/mcp", transport="http")
        mock_session = _make_mock_session(FAKE_TOOLS)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=_mock_streamablehttp_client,
            ),
            patch("mcp.client.session.ClientSession") as MockSession,
        ):
            MockSession.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client._http_list_tools()

        assert len(result) == 2
        names = {t["name"] for t in result}
        assert names == {"send_message", "list_channels"}

    @pytest.mark.asyncio
    async def test_http_call_tool_uses_streamablehttp_client(self):
        client = _TargetClient("https://server.com/mcp", transport="http")
        mock_session = _make_mock_session(FAKE_TOOLS)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=_mock_streamablehttp_client,
            ),
            patch("mcp.client.session.ClientSession") as MockSession,
        ):
            MockSession.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await client._http_call_tool("send_message", {"channel": "c", "text": "hi"})

        assert result  # non-empty response

    @pytest.mark.asyncio
    async def test_http_missing_mcp_library_raises_runtime_error(self, monkeypatch):
        client = _TargetClient("https://server.com/mcp", transport="http")

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mcp.client.streamable_http":
                raise ImportError("no streamable_http")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(RuntimeError, match="MCP library missing"):
            await client._http_list_tools()


# ─── _unwrap_exception_group ─────────────────────────────────────────────────

class TestUnwrapExceptionGroup:
    def test_plain_exception_returned_unchanged(self):
        exc = ValueError("plain error")
        assert _unwrap_exception_group(exc) is exc

    def test_exception_group_unwrapped_one_level(self):
        inner = RuntimeError("the real error")
        group = MagicMock()
        group.exceptions = [inner]
        result = _unwrap_exception_group(group)
        assert result is inner

    def test_exception_group_unwrapped_two_levels(self):
        deep = ConnectionError("connection refused")
        mid = MagicMock()
        mid.exceptions = [deep]
        outer = MagicMock()
        outer.exceptions = [mid]
        result = _unwrap_exception_group(outer)
        assert result is deep

    def test_no_exceptions_attribute_returned_unchanged(self):
        exc = TypeError("no group")
        assert _unwrap_exception_group(exc) is exc

    def test_empty_exceptions_list_returned_unchanged(self):
        group = MagicMock()
        group.exceptions = []
        result = _unwrap_exception_group(group)
        assert result is group


# ─── Compression is transport-agnostic ───────────────────────────────────────

class TestCompressionConsistency:
    """Verifies that schema compression produces identical output regardless of transport.

    Architecture note: compression lives entirely in RefractProxy.connect()
    (via build_index / compress_tool), not in the transport layer. These tests
    assert that property directly — same input tools → same compressed output.
    """

    def _compress_all(self, tools: list[dict]) -> dict[str, dict]:
        defs = collect_defs(tools)
        return {t["name"]: compress_tool(t, defs) for t in tools}

    def test_compression_output_is_deterministic(self):
        out1 = self._compress_all(FAKE_TOOLS)
        out2 = self._compress_all(FAKE_TOOLS)
        assert out1 == out2

    def test_compression_identical_for_all_transports(self, tmp_path):
        """Proxies using different transports but same tools produce identical compression."""
        schema_file = tmp_path / "tools.json"
        schema_file.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")

        def _make_proxy(transport):
            p = RefractProxy(str(schema_file), transport=transport)
            asyncio.run(p.connect())
            return p

        proxy_none = _make_proxy(None)
        proxy_sse = _make_proxy("sse")
        proxy_http = _make_proxy("http")

        # All three use the local JSON file (bypasses transport) — same tools loaded
        assert proxy_none._compressed == proxy_sse._compressed == proxy_http._compressed

    def test_compressed_schema_smaller_than_raw_for_http_path(self):
        from ast_extractor import count_tokens

        tool = FAKE_TOOLS[0]
        defs = collect_defs([tool])
        compressed = compress_tool(tool, defs)
        schema = _compressed_to_input_schema(compressed)

        raw_tok = count_tokens(json.dumps(tool["inputSchema"], ensure_ascii=False))
        comp_tok = count_tokens(json.dumps(schema, ensure_ascii=False))
        assert comp_tok <= raw_tok

    def test_proxy_with_http_transport_serves_same_tools_list(self, tmp_path):
        """tools/list output is identical whether transport is None, sse, or http."""
        schema_file = tmp_path / "tools.json"
        schema_file.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")

        def _tools_list(transport):
            p = RefractProxy(str(schema_file), transport=transport)
            asyncio.run(p.connect())
            return [(t.name, json.dumps(t.inputSchema)) for t in p.handle_tools_list()]

        result_none = _tools_list(None)
        result_http = _tools_list("http")
        result_sse = _tools_list("sse")

        assert result_none == result_http == result_sse


# ─── RefractProxy accepts transport param ────────────────────────────────────

class TestRefractProxyTransportParam:
    def test_proxy_stores_transport_in_client(self):
        proxy = RefractProxy("https://server.com/mcp", transport="http")
        assert proxy._client.transport == "http"

    def test_proxy_transport_none_by_default(self):
        proxy = RefractProxy("https://server.com")
        assert proxy._client.transport is None

    def test_proxy_sse_transport(self):
        proxy = RefractProxy("https://server.com/sse", transport="sse")
        assert proxy._client.transport == "sse"


# ─── main() integration: transport piped through ─────────────────────────────

class TestMainIntegrationTransport:
    def test_main_transport_http_passed_to_proxy(self, tmp_path, monkeypatch):
        schema_file = tmp_path / "t.json"
        schema_file.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")

        captured_transport = []

        import refract_proxy as rp

        original_init = rp.RefractProxy.__init__

        def patched_init(self, target_url, port=8080, verbose=False, use_cache=False, transport=None, sse_timeout=30.0):
            captured_transport.append(transport)
            original_init(self, target_url, port=port, verbose=verbose, use_cache=use_cache, transport=transport, sse_timeout=sse_timeout)

        async def mock_serve(self):
            pass

        monkeypatch.setattr(rp.RefractProxy, "__init__", patched_init)
        monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

        main(["--target", str(schema_file), "--transport", "stdio"])

        assert captured_transport == ["stdio"]

    def test_main_transport_http_requires_url_exits(self):
        with pytest.raises(SystemExit):
            main(["--stdio-cmd", "npx mcp-server", "--transport", "http"])
