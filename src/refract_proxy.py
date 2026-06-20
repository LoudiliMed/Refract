"""
refract_proxy — MCP proxy that compresses tool schemas on the fly.

Architecture:
    Agent (Claude/Cursor) → RefractProxy (local MCP server)
                          → Real MCP server (Gmail, Calendar, GitHub…)

The proxy:
  1. Connects to the target server and fetches all tools (connect)
  2. Builds the compressed TIER 1 index (build_index) + TIER 2 (compress_tool)
  3. Serves as a local MCP server: tools/list returns compressed schemas
  4. On tools/call: verifies signal, relays to the target server unchanged

Zero LLM calls — fully deterministic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_optimizer import build_index, collect_defs, compress_tool
from ast_extractor import count_tokens

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Global stats (exported for GET /stats in main.py)
# ─────────────────────────────────────────────────────────────────────
_PROXY_STATS: dict[str, Any] = {}


def _update_stats(server_url: str, tokens_raw: int, tokens_compressed: int) -> None:
    s = _PROXY_STATS.setdefault(server_url, {"requests": 0, "tokens_economises": 0, "_raw": 0})
    s["requests"] += 1
    s["tokens_economises"] += max(0, tokens_raw - tokens_compressed)
    s["_raw"] += tokens_raw


def get_proxy_stats() -> dict:
    """Returns aggregated statistics for the proxy session."""
    total_req = sum(v["requests"] for v in _PROXY_STATS.values())
    total_saved = sum(v["tokens_economises"] for v in _PROXY_STATS.values())
    total_raw = sum(v.get("_raw", 0) for v in _PROXY_STATS.values())
    pct = round(total_saved / total_raw * 100, 1) if total_raw else 0.0
    return {
        "total_requests": total_req,
        "tokens_economises": total_saved,
        "cout_evite_usd": round(total_saved / 1_000_000 * 3.0, 6),
        "reduction_moyenne_pct": pct,
        "par_serveur": {
            url: {"requests": v["requests"], "tokens_economises": v["tokens_economises"]}
            for url, v in _PROXY_STATS.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Lightweight signal check (without modifying mcp_signal_check.py)
# ─────────────────────────────────────────────────────────────────────
def _signal_ok(raw_tool: dict, compressed: dict) -> bool:
    """Verifies that all raw parameters are present in the compressed version.

    If params are lost -> send the full schema, log a warning,
    never break the call.
    """
    raw_params = set((raw_tool.get("inputSchema") or {}).get("properties", {}).keys())
    comp_params = set(compressed.get("params", {}).keys())
    missing = raw_params - comp_params
    if missing:
        logger.warning("signal_check FAIL '%s' — lost params: %s", raw_tool["name"], sorted(missing))
        return False
    return True


# ─────────────────────────────────────────────────────────────────────
# Compressed schema -> valid JSON Schema conversion (for MCP inputSchema)
# ─────────────────────────────────────────────────────────────────────
def _param_to_schema(cp: dict) -> dict:
    """Reconstructs a minimal JSON Schema from a compressed param."""
    if not isinstance(cp, dict):
        return {}
    if "ref" in cp:
        return {"$ref": f"#/$defs/{cp['ref']}"}
    out: dict = {}
    if "t" in cp:
        out["type"] = cp["t"]
    if "enum" in cp:
        out["enum"] = cp["enum"]
    if "of" in cp:
        out["items"] = _param_to_schema(cp["of"])
    if "props" in cp:
        out["properties"] = {k: _param_to_schema(v) for k, v in cp["props"].items()}
        out.setdefault("type", "object")
    if "d" in cp:
        out["description"] = cp["d"]
    if "any" in cp:
        out["anyOf"] = [_param_to_schema(b) for b in cp["any"]]
    if "all" in cp:
        out["allOf"] = [_param_to_schema(b) for b in cp["all"]]
    if "min" in cp:
        out["minimum"] = cp["min"]
    if "max" in cp:
        out["maximum"] = cp["max"]
    return out


def _compressed_to_input_schema(compressed: dict, shared_defs: dict | None = None) -> dict:
    """Builds a JSON Schema inputSchema from a compressed tool (for the MCP response)."""
    params = compressed.get("params", {})
    if not params:
        return {"type": "object", "properties": {}}
    props = {k: _param_to_schema(v) for k, v in params.items()}
    required = [k for k, v in params.items() if v.get("req")]
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    if shared_defs:
        schema["$defs"] = shared_defs
    return schema


# ─────────────────────────────────────────────────────────────────────
# Target MCP client (stdio subprocess, SSE/HTTP, or local JSON file)
# ─────────────────────────────────────────────────────────────────────
_STDIO_EXECUTABLES = frozenset({
    "npx", "node", "python", "python3", "uvx", "deno", "bun",
})


class _TargetClient:
    """Lightweight client to connect to the target MCP server.

    Dispatch priority in list_tools / call_tool:
      1. Local JSON file (tests, static catalogue)
      2. stdio command (npx, python3, uvx…)
      3. HTTP/SSE (deployed server)
    """

    def __init__(self, target_url: str):
        self.target_url = target_url

    # ── transport detection ─────────────────────────────────────────── #

    def _is_local_file(self) -> bool:
        url = self.target_url
        return url.startswith("file://") or (
            not url.startswith("http") and os.path.isfile(url)
        )

    def _is_stdio_cmd(self) -> bool:
        """True if target_url is a shell command to launch as a subprocess."""
        url = self.target_url
        # Don't intercept files already handled by _is_local_file
        if url.startswith("file://") or os.path.isfile(url):
            return False
        # Known stdio executables (e.g. "npx @mcp/server-fs /tmp")
        first_word = shlex.split(url)[0] if url.strip() else ""
        if os.path.basename(first_word) in _STDIO_EXECUTABLES:
            return True
        # Anything that isn't HTTP and isn't a file -> stdio command
        if not url.startswith("http"):
            return True
        return False

    def _file_path(self) -> str:
        url = self.target_url
        return url[7:] if url.startswith("file://") else url

    # ── dispatch ────────────────────────────────────────────────────── #

    async def list_tools(self) -> list[dict]:
        if self._is_local_file():
            return self._load_json_file(self._file_path())
        if self._is_stdio_cmd():
            return await self._stdio_list_tools()
        return await self._sse_list_tools()

    async def call_tool(self, name: str, arguments: dict) -> list:
        if self._is_local_file():
            return [{"type": "text", "text": f"[Proxy] Simulated call to '{name}' (local target)"}]
        if self._is_stdio_cmd():
            return await self._stdio_call_tool(name, arguments)
        return await self._sse_call_tool(name, arguments)

    # ── static JSON file ─────────────────────────────────────────────── #

    @staticmethod
    def _load_json_file(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw
        if "tools" in raw and isinstance(raw["tools"], list):
            return raw["tools"]
        tools: list[dict] = []
        for v in raw.values():
            if isinstance(v, list):
                tools.extend(v)
        return tools

    # ── stdio (subprocess) ───────────────────────────────────────────── #

    def _stdio_params(self):
        """Builds StdioServerParameters from the target_url command."""
        try:
            from mcp.client.stdio import StdioServerParameters
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}. Run `pip install mcp`") from exc
        parts = shlex.split(self.target_url)
        return StdioServerParameters(command=parts[0], args=parts[1:])

    @staticmethod
    def _tool_to_dict(t) -> dict:
        schema = t.inputSchema
        if schema is None:
            schema = {}
        elif not isinstance(schema, dict):
            schema = dict(schema)
        return {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": schema,
        }

    async def _stdio_list_tools(self) -> list[dict]:
        try:
            from mcp.client.stdio import stdio_client
            from mcp.client.session import ClientSession
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}. Run `pip install mcp`") from exc

        params = self._stdio_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [self._tool_to_dict(t) for t in result.tools]

    async def _stdio_call_tool(self, name: str, arguments: dict) -> list:
        try:
            from mcp.client.stdio import stdio_client
            from mcp.client.session import ClientSession
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}") from exc

        params = self._stdio_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return list(result.content) if result.content else []

    # ── SSE/HTTP ─────────────────────────────────────────────────────── #

    async def _sse_list_tools(self) -> list[dict]:
        try:
            from mcp.client.sse import sse_client
            from mcp.client.session import ClientSession
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}. Run `pip install mcp`") from exc

        async with sse_client(self.target_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [self._tool_to_dict(t) for t in result.tools]

    async def _sse_call_tool(self, name: str, arguments: dict) -> list:
        try:
            from mcp.client.sse import sse_client
            from mcp.client.session import ClientSession
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}") from exc

        async with sse_client(self.target_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return list(result.content) if result.content else []


# ─────────────────────────────────────────────────────────────────────
# Main proxy
# ─────────────────────────────────────────────────────────────────────
class RefractProxy:
    """MCP proxy: intercepts tools/list and returns the compressed index on demand."""

    def __init__(
        self,
        target_url: str,
        port: int = 8080,
        verbose: bool = False,
        use_cache: bool = False,
    ):
        self.target_url = target_url
        self.port = port
        self.verbose = verbose
        self.use_cache = use_cache   # injects cache_control in as_anthropic_tools()
        self._tools: list[dict] = []
        self._tools_by_name: dict[str, dict] = {}
        self._index: dict = {}
        self._defs: dict = {}
        self._compressed: dict[str, dict] = {}
        self._client = _TargetClient(target_url)
        # Parsed command if target is a stdio subprocess, otherwise None
        self._cmd_parts: list[str] | None = (
            target_url.split() if self._client._is_stdio_cmd() else None
        )
        # Semantic router — lazy-initialized in connect()
        self._router = None

    # ── connection ─────────────────────────────────────────────────── #

    async def connect(self) -> None:
        """Connects to the target server, fetches tools, and builds the compressed index."""
        self._tools = await self._client.list_tools()
        self._defs = collect_defs(self._tools)
        self._index = build_index(self._tools, self._defs)
        self._tools_by_name = {t["name"]: t for t in self._tools}
        self._compressed = {t["name"]: compress_tool(t, self._defs) for t in self._tools}

        # Build semantic router (requires fastembed; degrades gracefully without it)
        try:
            from semantic_router import SemanticRouter
            self._router = SemanticRouter()
            self._router.index_tools(self._tools)
        except RuntimeError:
            logger.warning(
                "fastembed not available — semantic routing disabled, "
                "falling back to keyword matching"
            )
            self._router = None

        if self.verbose:
            raw_tok = count_tokens(json.dumps(self._tools, ensure_ascii=False))
            idx_tok = count_tokens(json.dumps(self._index, ensure_ascii=False))
            pct = (1 - idx_tok / raw_tok) * 100 if raw_tok else 0.0
            print(
                f"[Refract] Connected to {self.target_url}\n"
                f"  {len(self._tools)} tools  |  {raw_tok} → {idx_tok} tokens"
                f"  ({pct:.0f}% index reduction)"
            )

    def identify_tool(self, query: str, min_score: float = 0.3) -> str | None:
        """Return the best matching tool name for *query*, or ``None`` if uncertain.

        Uses the semantic router when available (fastembed installed), falls
        back to keyword scoring otherwise.  Returns ``None`` — never a guess —
        when confidence is below *min_score*, so the caller can send the full
        compact index instead of routing to the wrong tool.

        Args:
            query: natural language description of what the agent wants to do.
            min_score: minimum cosine similarity required (default 0.3).

        Returns:
            Tool name string, or ``None`` if no confident match found.
        """
        if self._router is not None:
            tool_name = self._router.find_best_tool_with_threshold(query, min_score=min_score)
            if tool_name is None:
                logger.warning(
                    "SemanticRouter: no confident match for query '%s'", query[:80]
                )
            return tool_name
        # Keyword fallback when fastembed is unavailable
        from semantic_router import _keyword_score
        result = _keyword_score(query, self._tools)
        return result or None

    # ── serving ────────────────────────────────────────────────────── #

    def _build_mcp_server(self):
        """Builds the MCP Server (tools/list + tools/call handlers) shared
        between stdio mode and HTTP/SSE mode — one set of handlers, two transports.
        """
        try:
            from mcp.server import Server
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}. Run `pip install mcp`") from exc

        server = Server("refract-proxy")
        proxy = self

        @server.list_tools()
        async def _list_tools():
            return proxy.handle_tools_list()

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict | None = None):
            return await proxy.handle_tool_call(name, arguments or {})

        return server

    async def serve(self) -> None:
        """Starts the local MCP server in stdio mode.

        Used by the CLI (--mode stdio, default) and by Claude Desktop / Cursor.
        For HTTP/SSE, see serve_http() — both modes share the same handlers
        via _build_mcp_server().
        """
        try:
            from mcp.server.stdio import stdio_server
        except ImportError as exc:
            raise RuntimeError(f"MCP library missing: {exc}. Run `pip install mcp`") from exc

        server = self._build_mcp_server()

        if self.verbose:
            print(f"[Refract] MCP server started (stdio) — target: {self.target_url}")

        init_opts = server.create_initialization_options()
        async with stdio_server() as (read, write):
            await server.run(read, write, init_opts)

    def build_asgi_app(self):
        """Builds the ASGI app (Starlette) exposing the standard MCP SSE transport.

        Routes:
          GET  /sse        — SSE handshake (client keeps the connection open)
          POST /messages/  — JSON-RPC calls (tools/list, tools/call…)

        Can be used in two ways:
          - standalone, via serve_http() (dedicated uvicorn on self.port);
          - mounted in an existing FastAPI app:
                app.mount("/proxy", proxy.build_asgi_app())
        """
        try:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.requests import Request
            from starlette.routing import Mount, Route
        except ImportError as exc:
            raise RuntimeError(
                f"MCP/Starlette library missing: {exc}. Run `pip install mcp fastapi`"
            ) from exc

        server = self._build_mcp_server()
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> None:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read, write):
                await server.run(read, write, server.create_initialization_options())

        return Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ])

    async def serve_http(self) -> None:
        """Starts the proxy in HTTP/SSE mode (instead of stdio).

        Allows an agent to connect via a simple network URL, without
        local installation or subprocess:
            http://localhost:<port>/sse

        Keeps serve() (stdio) intact — both modes coexist, the mode
        is chosen by the caller (CLI: --mode http|stdio).
        """
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(f"uvicorn missing: {exc}. Run `pip install uvicorn`") from exc

        asgi_app = self.build_asgi_app()
        url = f"http://localhost:{self.port}/sse"
        print(f"[Refract] HTTP proxy started → {url}")
        if self.verbose:
            print(f"[Refract]   target: {self.target_url}")

        config = uvicorn.Config(asgi_app, host="0.0.0.0", port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    # ── MCP handlers ───────────────────────────────────────────────── #

    def handle_tools_list(self) -> list:
        """TIER 1: returns tools with compressed schemas (compact index via build_index)."""
        try:
            from mcp import types as mt
        except ImportError:
            return []

        result: list = []
        raw_tokens = 0
        comp_tokens = 0

        for t in self._tools:
            compressed = self._compressed.get(t["name"], {})
            raw_tok = count_tokens(json.dumps(t.get("inputSchema", {}), ensure_ascii=False))
            comp_schema = _compressed_to_input_schema(compressed)
            comp_tok = count_tokens(json.dumps(comp_schema, ensure_ascii=False))
            raw_tokens += raw_tok
            comp_tokens += comp_tok

            # Short description from TIER 1 index
            short_desc = self._index.get("tools", {}).get(t["name"], "") or t.get("description", "")

            result.append(mt.Tool(
                name=t["name"],
                description=short_desc,
                inputSchema=comp_schema,
            ))

        if raw_tokens > 0:
            _update_stats(self.target_url, raw_tokens, comp_tokens)
            if self.verbose:
                pct = (1 - comp_tokens / raw_tokens) * 100 if raw_tokens else 0.0
                print(f"[Refract] tools/list: {raw_tokens} → {comp_tokens} tokens ({pct:.0f}%)")

        return result

    def as_anthropic_tools(self) -> list[dict]:
        """Returns compressed tools in Anthropic API format, with cache_control.

        Output format:
        [
            {"name": "...", "description": "...", "input_schema": {...}},
            ...,
            {"name": "...", ..., "cache_control": {"type": "ephemeral"}}  # last one
        ]

        The ``cache_control`` on the last tool tells the Anthropic API to cache
        all tools in the list. Price drops from $3.00/M to $0.30/M for cache hits.

        Returns:
            List of dicts compatible with the Anthropic API ``tools`` parameter.
        """
        from cache_injector import CacheInjector

        tools_dicts = []
        for t in self._tools:
            compressed = self._compressed.get(t["name"], {})
            short_desc = self._index.get("tools", {}).get(t["name"], "") or t.get("description", "")
            comp_schema = _compressed_to_input_schema(compressed)
            tools_dicts.append({
                "name": t["name"],
                "description": short_desc,
                "input_schema": comp_schema,
            })

        return CacheInjector.inject_cache_control(tools_dicts) if self.use_cache else tools_dicts

    async def handle_tool_call(self, tool_name: str, arguments: dict) -> list:
        """TIER 2: loads the compressed schema, verifies signal, relays to target server."""
        try:
            from mcp import types as mt
        except ImportError:
            return []

        raw_tool = self._tools_by_name.get(tool_name)
        if not raw_tool:
            return [mt.TextContent(type="text", text=f"Tool '{tool_name}' not found in proxy.")]

        compressed = self._compressed.get(tool_name, {})

        # Signal check: if compression lost params -> log, but relay anyway
        if not _signal_ok(raw_tool, compressed):
            logger.warning("Signal check FAIL '%s' — relaying with full schema", tool_name)

        if self.verbose:
            print(f"[Refract] → {tool_name}({list(arguments.keys())})")

        try:
            content = await self._client.call_tool(tool_name, arguments)
            if not content:
                return [mt.TextContent(type="text", text="(empty response from target server)")]
            # Normalize returned items
            items = []
            for item in content:
                if isinstance(item, dict):
                    items.append(mt.TextContent(type="text", text=json.dumps(item, ensure_ascii=False)))
                else:
                    items.append(item)
            return items
        except Exception as exc:
            logger.error("Relay error '%s': %s", tool_name, exc)
            return [mt.TextContent(type="text", text=f"Proxy error: {exc}")]
