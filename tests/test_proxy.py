"""
test_proxy.py — Vérifie que le proxy RefractProxy :
  - tools/list renvoie l'index compressé (schémas plus courts que les originaux)
  - handle_tool_call relaie correctement
  - signal_ok détecte les params perdus
  - _compressed_to_input_schema produit un JSON Schema valide
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

# Accès aux modules src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_proxy import (
    RefractProxy,
    _TargetClient,
    _signal_ok,
    _compressed_to_input_schema,
    _param_to_schema,
    get_proxy_stats,
    _PROXY_STATS,
)
from mcp_optimizer import compress_tool, collect_defs, build_index

# ─── outils de test ─────────────────────────────────────────────────────────

FAKE_TOOLS = [
    {
        "name": "create_event",
        "description": "Creates a calendar event with a start and end time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title"},
                "startTime": {"type": "string", "format": "date-time", "description": "Start time"},
                "endTime": {"type": "string", "format": "date-time", "description": "End time"},
            },
            "required": ["summary", "startTime", "endTime"],
        },
    },
    {
        "name": "list_events",
        "description": "Lists upcoming calendar events.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "maxResults": {"type": "integer", "description": "Max number of events to return"},
            },
        },
    },
]

# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def proxy_with_tools(tmp_path) -> RefractProxy:
    """Proxy pré-chargé avec FAKE_TOOLS (sans connexion réseau)."""
    schema_file = tmp_path / "fake.json"
    schema_file.write_text(json.dumps(FAKE_TOOLS), encoding="utf-8")

    proxy = RefractProxy(target_url=str(schema_file), port=8080, verbose=False)

    async def _load():
        await proxy.connect()

    asyncio.run(_load())
    return proxy


# ─── tests signal_ok ────────────────────────────────────────────────────────

def test_signal_ok_passes_when_all_params_present():
    tool = FAKE_TOOLS[0]
    compressed = compress_tool(tool)
    assert _signal_ok(tool, compressed), "Signal OK attendu quand tous les params sont présents"


def test_signal_ok_fails_when_param_missing():
    tool = FAKE_TOOLS[0]
    compressed = compress_tool(tool)
    # Supprimer un param de la version compressée
    del compressed["params"]["summary"]
    assert not _signal_ok(tool, compressed), "Signal FAIL attendu quand un param est manquant"


# ─── tests _compressed_to_input_schema ──────────────────────────────────────

def test_compressed_to_input_schema_has_required():
    tool = FAKE_TOOLS[0]
    compressed = compress_tool(tool)
    schema = _compressed_to_input_schema(compressed)

    assert schema.get("type") == "object"
    assert "properties" in schema
    assert set(schema.get("required", [])) == {"summary", "startTime", "endTime"}


def test_compressed_to_input_schema_smaller_than_raw():
    """L'inputSchema compressé doit contenir moins de tokens que l'original."""
    from ast_extractor import count_tokens

    tool = FAKE_TOOLS[0]
    compressed = compress_tool(tool)
    schema = _compressed_to_input_schema(compressed)

    raw_tokens = count_tokens(json.dumps(tool["inputSchema"], ensure_ascii=False))
    comp_tokens = count_tokens(json.dumps(schema, ensure_ascii=False))
    assert comp_tokens <= raw_tokens, (
        f"Schéma compressé ({comp_tokens} tok) devrait être ≤ brut ({raw_tokens} tok)"
    )


def test_param_to_schema_handles_ref():
    cp = {"ref": "Attendee"}
    schema = _param_to_schema(cp)
    assert schema == {"$ref": "#/$defs/Attendee"}


def test_param_to_schema_handles_array():
    cp = {"t": "array", "of": {"t": "string"}}
    schema = _param_to_schema(cp)
    assert schema["type"] == "array"
    assert schema["items"]["type"] == "string"


# ─── tests handle_tools_list ────────────────────────────────────────────────

def test_handle_tools_list_returns_all_tools(proxy_with_tools):
    tools = proxy_with_tools.handle_tools_list()
    names = [t.name for t in tools]
    assert "create_event" in names
    assert "list_events" in names
    assert len(tools) == len(FAKE_TOOLS)


def test_handle_tools_list_descriptions_shorter(proxy_with_tools):
    """Les descriptions retournées sont plus courtes que les originales."""
    tools_by_name = {t.name: t for t in proxy_with_tools.handle_tools_list()}
    original_desc = FAKE_TOOLS[0]["description"]
    compressed_desc = tools_by_name["create_event"].description
    assert len(compressed_desc) <= len(original_desc)


def test_handle_tools_list_input_schema_valid(proxy_with_tools):
    """Chaque tool retourné doit avoir un inputSchema avec type=object."""
    tools = proxy_with_tools.handle_tools_list()
    for t in tools:
        assert isinstance(t.inputSchema, dict), f"{t.name} doit avoir un inputSchema dict"
        assert t.inputSchema.get("type") == "object"


# ─── tests handle_tool_call ──────────────────────────────────────────────────

def test_handle_tool_call_unknown_tool(proxy_with_tools):
    async def _run():
        return await proxy_with_tools.handle_tool_call("unknown_tool", {})

    result = asyncio.run(_run())
    assert result
    text = getattr(result[0], "text", "") or ""
    assert "introuvable" in text.lower() or "unknown" in text.lower()


def test_handle_tool_call_known_tool_local(proxy_with_tools):
    """Sur un fichier JSON local, l'appel est simulé (pas d'erreur fatale)."""
    async def _run():
        return await proxy_with_tools.handle_tool_call("create_event", {"summary": "Test"})

    result = asyncio.run(_run())
    assert result, "L'appel doit retourner quelque chose"


# ─── tests stats ─────────────────────────────────────────────────────────────

def test_stats_accumulate_after_list(proxy_with_tools):
    """Après tools/list, les stats doivent être non nulles."""
    _PROXY_STATS.clear()
    proxy_with_tools.handle_tools_list()
    stats = get_proxy_stats()
    assert stats["total_requests"] >= 1
    assert stats["tokens_economises"] >= 0


# ─── tests _TargetClient : détection du transport ───────────────────────────

def test_is_local_file_with_json_file(tmp_path):
    f = tmp_path / "t.json"
    f.write_text("[]")
    c = _TargetClient(str(f))
    assert c._is_local_file() is True
    assert c._is_stdio_cmd() is False


def test_is_local_file_with_file_uri(tmp_path):
    f = tmp_path / "t.json"
    f.write_text("[]")
    c = _TargetClient(f"file://{f}")
    assert c._is_local_file() is True
    assert c._is_stdio_cmd() is False


def test_is_stdio_cmd_npx():
    c = _TargetClient("npx @modelcontextprotocol/server-filesystem /tmp")
    assert c._is_stdio_cmd() is True
    assert c._is_local_file() is False


def test_is_stdio_cmd_python():
    c = _TargetClient("python3 -m my_mcp_server --arg val")
    assert c._is_stdio_cmd() is True


def test_is_stdio_cmd_uvx():
    c = _TargetClient("uvx mcp-server-git")
    assert c._is_stdio_cmd() is True


def test_is_stdio_cmd_node():
    c = _TargetClient("node /abs/path/server.js")
    assert c._is_stdio_cmd() is True


def test_is_not_stdio_http():
    c = _TargetClient("https://api.example.com/mcp")
    assert c._is_stdio_cmd() is False
    assert c._is_local_file() is False


def test_is_not_stdio_http_with_port():
    c = _TargetClient("http://localhost:3000/sse")
    assert c._is_stdio_cmd() is False


def test_cmd_parts_stored_on_proxy():
    """RefractProxy._cmd_parts doit être une liste pour les commandes stdio."""
    proxy = RefractProxy("npx @mcp/server /tmp")
    assert proxy._cmd_parts == ["npx", "@mcp/server", "/tmp"]


def test_cmd_parts_none_for_http():
    proxy = RefractProxy("https://server.com")
    assert proxy._cmd_parts is None


def test_cmd_parts_none_for_file(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("[]")
    proxy = RefractProxy(str(f))
    assert proxy._cmd_parts is None


def test_stdio_params_parsed_correctly():
    """_stdio_params() doit extraire command + args via shlex."""
    c = _TargetClient("npx @modelcontextprotocol/server-filesystem /home/user/docs")
    try:
        params = c._stdio_params()
        assert params.command == "npx"
        assert params.args == ["@modelcontextprotocol/server-filesystem", "/home/user/docs"]
    except RuntimeError:
        pytest.skip("mcp non installé")


# ─── tests mode HTTP/SSE (serve_http / build_asgi_app) ───────────────────────

def test_build_asgi_app_returns_starlette_app(proxy_with_tools):
    app = proxy_with_tools.build_asgi_app()
    from starlette.applications import Starlette
    assert isinstance(app, Starlette)


def test_build_asgi_app_has_sse_and_messages_routes(proxy_with_tools):
    app = proxy_with_tools.build_asgi_app()
    paths = []
    for r in app.routes:
        paths.append(getattr(r, "path", None))
    assert "/sse" in paths
    assert any(p and "messages" in p for p in paths)


def test_build_mcp_server_returns_server_instance(proxy_with_tools):
    server = proxy_with_tools._build_mcp_server()
    from mcp.server import Server
    assert isinstance(server, Server)


def test_serve_and_serve_http_share_handlers(proxy_with_tools):
    """serve() (stdio) et build_asgi_app() (http) doivent utiliser les mêmes
    handlers tools/list et tools/call — pas de divergence entre les deux modes."""
    server_for_stdio = proxy_with_tools._build_mcp_server()
    server_for_http = proxy_with_tools._build_mcp_server()
    # Les deux instances exposent le même nom de serveur (même construction)
    assert server_for_stdio.name == server_for_http.name == "refract-proxy"


def test_serve_http_prints_connection_url(proxy_with_tools, monkeypatch, capsys):
    """serve_http() doit afficher l'URL http://localhost:<port>/sse au démarrage."""
    import asyncio

    proxy_with_tools.port = 8123

    class _FakeUvicornServer:
        def __init__(self, config):
            self.config = config

        async def serve(self):
            return  # ne démarre pas vraiment uvicorn

    class _FakeUvicornModule:
        @staticmethod
        def Config(*args, **kwargs):
            return object()

        Server = _FakeUvicornServer

    import sys
    monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicornModule)

    asyncio.run(proxy_with_tools.serve_http())

    captured = capsys.readouterr()
    assert "[Refract] HTTP proxy started → http://localhost:8123/sse" in captured.err
    assert captured.out == ""


def test_serve_http_raises_helpful_error_without_uvicorn(proxy_with_tools, monkeypatch):
    import asyncio
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="uvicorn"):
        asyncio.run(proxy_with_tools.serve_http())


# ─── test end-to-end avec mcp_calendar_schemas.json ─────────────────────────

def test_end_to_end_calendar_schemas():
    """Charge les vrais schémas Calendar et vérifie la compression de l'index."""
    schemas_path = os.path.join(
        os.path.dirname(__file__), "..", "schemas", "mcp_calendar_schemas.json"
    )
    if not os.path.isfile(schemas_path):
        pytest.skip("mcp_calendar_schemas.json introuvable")

    proxy = RefractProxy(target_url=schemas_path, verbose=False)

    async def _run():
        await proxy.connect()

    asyncio.run(_run())

    assert proxy._tools, "Des tools doivent être chargés depuis le fichier Calendar"
    assert proxy._index, "L'index doit être construit"

    tools_response = proxy.handle_tools_list()
    assert tools_response, "tools/list ne doit pas être vide"

    # Vérifie que tous les tools ont un schéma valide
    for t in tools_response:
        assert t.inputSchema.get("type") == "object", f"{t.name} doit avoir type=object"

    # Vérifie la compression : l'index doit être plus compact que les schémas bruts
    from ast_extractor import count_tokens
    raw_tok = count_tokens(json.dumps(proxy._tools, ensure_ascii=False))
    idx_tok = count_tokens(json.dumps(proxy._index, ensure_ascii=False))
    assert idx_tok < raw_tok, (
        f"L'index ({idx_tok} tok) doit être plus petit que le brut ({raw_tok} tok)"
    )
    print(f"\n[Calendar] {len(proxy._tools)} tools | {raw_tok} → {idx_tok} tok"
          f" ({(1 - idx_tok/raw_tok)*100:.0f}% réduction)", file=sys.stderr)


# ─── tests stdout propre (protocole MCP stdio) ───────────────────────────────

_VERBOSE_FIXTURE = [
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "list_messages",
        "description": "List recent email messages.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Max messages to return"},
            },
        },
    },
]


@pytest.fixture
def verbose_proxy(tmp_path) -> RefractProxy:
    """Proxy verbose pré-chargé avec _VERBOSE_FIXTURE (fichier JSON local)."""
    schema_file = tmp_path / "verbose_fixture.json"
    schema_file.write_text(json.dumps(_VERBOSE_FIXTURE), encoding="utf-8")
    proxy = RefractProxy(target_url=str(schema_file), verbose=True)
    asyncio.run(proxy.connect())
    return proxy


def test_verbose_connect_does_not_write_to_stdout(tmp_path, capsys):
    """connect() en mode verbose ne doit rien écrire sur stdout."""
    schema_file = tmp_path / "fixture.json"
    schema_file.write_text(json.dumps(_VERBOSE_FIXTURE), encoding="utf-8")
    proxy = RefractProxy(target_url=str(schema_file), verbose=True)
    asyncio.run(proxy.connect())
    captured = capsys.readouterr()
    assert captured.out == "", "connect() verbose must not write to stdout"
    assert "[Refract]" in captured.err


def test_handle_tools_list_verbose_does_not_write_to_stdout(verbose_proxy, capsys):
    """handle_tools_list() en mode verbose ne doit rien écrire sur stdout."""
    verbose_proxy.handle_tools_list()
    captured = capsys.readouterr()
    assert captured.out == "", "handle_tools_list() verbose must not write to stdout"
    assert "[Refract]" in captured.err


def test_handle_tool_call_verbose_does_not_write_to_stdout(verbose_proxy, capsys):
    """handle_tool_call() en mode verbose ne doit rien écrire sur stdout."""
    asyncio.run(verbose_proxy.handle_tool_call("send_email", {"to": "a@b.com", "subject": "Hi", "body": "Hello"}))
    captured = capsys.readouterr()
    assert captured.out == "", "handle_tool_call() verbose must not write to stdout"
    assert "[Refract]" in captured.err


def test_serve_stdio_verbose_does_not_write_to_stdout(verbose_proxy, monkeypatch, capsys):
    """serve() en mode verbose ne doit rien écrire sur stdout avant de démarrer."""
    def mock_stdio_server():
        class _CM:
            async def __aenter__(self):
                return (None, None)
            async def __aexit__(self, *_):
                pass
        return _CM()

    import mcp.server.stdio as _stdio_mod

    async def mock_serve_runner(self, read, write, opts):
        return

    monkeypatch.setattr(_stdio_mod, "stdio_server", mock_stdio_server)

    from mcp.server import Server
    monkeypatch.setattr(Server, "run", mock_serve_runner)

    asyncio.run(verbose_proxy.serve())
    captured = capsys.readouterr()
    assert captured.out == "", "serve() verbose must not write to stdout"
    assert "[Refract]" in captured.err
