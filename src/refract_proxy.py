"""
refract_proxy — Proxy MCP qui compresse les schémas de tools à la volée.

Architecture :
    Agent (Claude/Cursor) → RefractProxy (serveur MCP local)
                          → Serveur MCP réel (Gmail, Calendar, GitHub…)

Le proxy :
  1. Se connecte au serveur cible et récupère tous les tools (connect)
  2. Construit l'index compressé TIER 1 (build_index) + TIER 2 (compress_tool)
  3. Sert comme serveur MCP local : tools/list renvoie les schémas compressés
  4. Sur tools/call : vérifie le signal, relaie au serveur cible sans modification

Zéro appel LLM — tout déterministe.
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
# Stats globales (exportées pour GET /stats dans main.py)
# ─────────────────────────────────────────────────────────────────────
_PROXY_STATS: dict[str, Any] = {}


def _update_stats(server_url: str, tokens_raw: int, tokens_compressed: int) -> None:
    s = _PROXY_STATS.setdefault(server_url, {"requests": 0, "tokens_economises": 0, "_raw": 0})
    s["requests"] += 1
    s["tokens_economises"] += max(0, tokens_raw - tokens_compressed)
    s["_raw"] += tokens_raw


def get_proxy_stats() -> dict:
    """Retourne les statistiques agrégées de la session proxy."""
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
# Signal check léger (sans modifier mcp_signal_check.py)
# ─────────────────────────────────────────────────────────────────────
def _signal_ok(raw_tool: dict, compressed: dict) -> bool:
    """Vérifie que tous les paramètres bruts sont présents dans la version compressée.

    Si des params sont perdus → renvoyer le schéma complet, logger un warning,
    ne jamais casser l'appel.
    """
    raw_params = set((raw_tool.get("inputSchema") or {}).get("properties", {}).keys())
    comp_params = set(compressed.get("params", {}).keys())
    missing = raw_params - comp_params
    if missing:
        logger.warning("signal_check FAIL '%s' — params perdus : %s", raw_tool["name"], sorted(missing))
        return False
    return True


# ─────────────────────────────────────────────────────────────────────
# Conversion schéma compressé → JSON Schema valide (pour inputSchema MCP)
# ─────────────────────────────────────────────────────────────────────
def _param_to_schema(cp: dict) -> dict:
    """Reconstruit un JSON Schema minimal depuis un param compressé."""
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
    """inputSchema JSON Schema depuis un tool compressé (pour la réponse MCP)."""
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
# Client MCP cible (stdio subprocess, SSE/HTTP ou fichier JSON local)
# ─────────────────────────────────────────────────────────────────────
_STDIO_EXECUTABLES = frozenset({
    "npx", "node", "python", "python3", "uvx", "deno", "bun",
})


class _TargetClient:
    """Client léger pour se connecter au serveur MCP cible.

    Priorité de dispatch dans list_tools / call_tool :
      1. Fichier JSON local (tests, catalogue statique)
      2. Commande stdio (npx, python3, uvx …)
      3. HTTP/SSE (serveur déployé)
    """

    def __init__(self, target_url: str):
        self.target_url = target_url

    # ── détection du transport ──────────────────────────────────────── #

    def _is_local_file(self) -> bool:
        url = self.target_url
        return url.startswith("file://") or (
            not url.startswith("http") and os.path.isfile(url)
        )

    def _is_stdio_cmd(self) -> bool:
        """True si target_url est une commande shell à lancer en sous-processus."""
        url = self.target_url
        # Ne pas intercepter les fichiers déjà gérés par _is_local_file
        if url.startswith("file://") or os.path.isfile(url):
            return False
        # Exécutables stdio connus (ex : "npx @mcp/server-fs /tmp")
        first_word = shlex.split(url)[0] if url.strip() else ""
        if os.path.basename(first_word) in _STDIO_EXECUTABLES:
            return True
        # Tout ce qui n'est pas HTTP et n'est pas un fichier → commande stdio
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
            return [{"type": "text", "text": f"[Proxy] Appel simulé de '{name}' (cible locale)"}]
        if self._is_stdio_cmd():
            return await self._stdio_call_tool(name, arguments)
        return await self._sse_call_tool(name, arguments)

    # ── fichier JSON statique ────────────────────────────────────────── #

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

    # ── stdio (sous-processus) ───────────────────────────────────────── #

    def _stdio_params(self):
        """Construit StdioServerParameters depuis la commande target_url."""
        try:
            from mcp.client.stdio import StdioServerParameters
        except ImportError as exc:
            raise RuntimeError(f"Lib MCP manquante : {exc}. Faites `pip install mcp`") from exc
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
            raise RuntimeError(f"Lib MCP manquante : {exc}. Faites `pip install mcp`") from exc

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
            raise RuntimeError(f"Lib MCP manquante : {exc}") from exc

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
            raise RuntimeError(f"Lib MCP manquante : {exc}. Faites `pip install mcp`") from exc

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
            raise RuntimeError(f"Lib MCP manquante : {exc}") from exc

        async with sse_client(self.target_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return list(result.content) if result.content else []


# ─────────────────────────────────────────────────────────────────────
# Proxy principal
# ─────────────────────────────────────────────────────────────────────
class RefractProxy:
    """Proxy MCP : intercepte tools/list et renvoie l'index compressé à la demande."""

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
        self.use_cache = use_cache   # injecte cache_control dans as_anthropic_tools()
        self._tools: list[dict] = []
        self._tools_by_name: dict[str, dict] = {}
        self._index: dict = {}
        self._defs: dict = {}
        self._compressed: dict[str, dict] = {}
        self._client = _TargetClient(target_url)
        # Commande parsée si la cible est un sous-processus stdio, sinon None
        self._cmd_parts: list[str] | None = (
            target_url.split() if self._client._is_stdio_cmd() else None
        )

    # ── connexion ──────────────────────────────────────────────────── #

    async def connect(self) -> None:
        """Connexion au serveur cible, récupère les tools et construit l'index compressé."""
        self._tools = await self._client.list_tools()
        self._defs = collect_defs(self._tools)
        self._index = build_index(self._tools, self._defs)
        self._tools_by_name = {t["name"]: t for t in self._tools}
        self._compressed = {t["name"]: compress_tool(t, self._defs) for t in self._tools}

        if self.verbose:
            raw_tok = count_tokens(json.dumps(self._tools, ensure_ascii=False))
            idx_tok = count_tokens(json.dumps(self._index, ensure_ascii=False))
            pct = (1 - idx_tok / raw_tok) * 100 if raw_tok else 0.0
            print(
                f"[Refract] Connecté à {self.target_url}\n"
                f"  {len(self._tools)} tools  |  {raw_tok} → {idx_tok} tokens"
                f"  ({pct:.0f}% réduction index)"
            )

    # ── serving ────────────────────────────────────────────────────── #

    def _build_mcp_server(self):
        """Construit le Server MCP (handlers tools/list + tools/call) partagé
        entre le mode stdio et le mode HTTP/SSE — un seul jeu de handlers,
        deux transports possibles.
        """
        try:
            from mcp.server import Server
        except ImportError as exc:
            raise RuntimeError(f"Lib MCP manquante : {exc}. Faites `pip install mcp`") from exc

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
        """Démarre le serveur MCP local en mode stdio.

        Utilisé par le CLI (--mode stdio, défaut) et par Claude Desktop / Cursor.
        Pour HTTP/SSE, voir serve_http() — les deux modes coexistent et
        partagent les mêmes handlers via _build_mcp_server().
        """
        try:
            from mcp.server.stdio import stdio_server
        except ImportError as exc:
            raise RuntimeError(f"Lib MCP manquante : {exc}. Faites `pip install mcp`") from exc

        server = self._build_mcp_server()

        if self.verbose:
            print(f"[Refract] Serveur MCP démarré (stdio) — cible : {self.target_url}")

        init_opts = server.create_initialization_options()
        async with stdio_server() as (read, write):
            await server.run(read, write, init_opts)

    def build_asgi_app(self):
        """Construit l'app ASGI (Starlette) exposant le transport SSE MCP standard.

        Routes :
          GET  /sse        — poignée de main SSE (le client garde la connexion ouverte)
          POST /messages/  — appels JSON-RPC (tools/list, tools/call…)

        Réutilisable de deux façons :
          - en standalone, via serve_http() (uvicorn dédié sur self.port) ;
          - montée dans une app FastAPI existante :
                app.mount("/proxy", proxy.build_asgi_app())
        """
        try:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.requests import Request
            from starlette.routing import Mount, Route
        except ImportError as exc:
            raise RuntimeError(
                f"Lib MCP/Starlette manquante : {exc}. Faites `pip install mcp fastapi`"
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
        """Démarre le proxy en mode HTTP/SSE (au lieu de stdio).

        Permet à un agent de se connecter via une simple URL réseau, sans
        installation locale ni sous-processus :
            http://localhost:<port>/sse

        Garde serve() (stdio) intact — les deux modes coexistent, le mode
        est choisi par l'appelant (CLI : --mode http|stdio).
        """
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(f"uvicorn manquant : {exc}. Faites `pip install uvicorn`") from exc

        asgi_app = self.build_asgi_app()
        url = f"http://localhost:{self.port}/sse"
        print(f"[Refract] Proxy HTTP démarré → {url}")
        if self.verbose:
            print(f"[Refract]   cible : {self.target_url}")

        config = uvicorn.Config(asgi_app, host="0.0.0.0", port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    # ── handlers MCP ───────────────────────────────────────────────── #

    def handle_tools_list(self) -> list:
        """TIER 1 : renvoie les tools avec schémas compressés (index compact via build_index)."""
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

            # Description courte depuis l'index TIER 1
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
                print(f"[Refract] tools/list : {raw_tokens} → {comp_tokens} tokens ({pct:.0f}%)")

        return result

    def as_anthropic_tools(self) -> list[dict]:
        """Retourne les tools compressés au format Anthropic API, avec cache_control.

        Format de sortie :
        [
            {"name": "...", "description": "...", "input_schema": {...}},
            ...,
            {"name": "...", ..., "cache_control": {"type": "ephemeral"}}  # dernier
        ]

        Le ``cache_control`` sur le dernier tool indique à l'API Anthropic de
        mettre en cache tous les tools de cette liste. Prix réduit de 3,00 $/M
        à 0,30 $/M pour les hits suivants.

        Returns:
            Liste de dicts compatibles avec le paramètre ``tools`` de l'API Anthropic.
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
        """TIER 2 : charge le schéma compressé, vérifie le signal, relaie au serveur cible."""
        try:
            from mcp import types as mt
        except ImportError:
            return []

        raw_tool = self._tools_by_name.get(tool_name)
        if not raw_tool:
            return [mt.TextContent(type="text", text=f"Tool '{tool_name}' introuvable dans le proxy.")]

        compressed = self._compressed.get(tool_name, {})

        # Signal check : si compression a perdu des params → on log, mais on relaie quand même
        if not _signal_ok(raw_tool, compressed):
            logger.warning("Signal check FAIL '%s' — appel relayé avec schéma complet", tool_name)

        if self.verbose:
            print(f"[Refract] → {tool_name}({list(arguments.keys())})")

        try:
            content = await self._client.call_tool(tool_name, arguments)
            if not content:
                return [mt.TextContent(type="text", text="(réponse vide du serveur cible)")]
            # Normalise les éléments retournés
            items = []
            for item in content:
                if isinstance(item, dict):
                    items.append(mt.TextContent(type="text", text=json.dumps(item, ensure_ascii=False)))
                else:
                    items.append(item)
            return items
        except Exception as exc:
            logger.error("Erreur relai '%s': %s", tool_name, exc)
            return [mt.TextContent(type="text", text=f"Erreur proxy : {exc}")]
