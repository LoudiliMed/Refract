"""
test_cli.py — Vérifie que le CLI refract_cli parse les args et configure le proxy.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refract_cli import _build_parser, main


# ─── tests du parser ─────────────────────────────────────────────────────────

def test_parser_requires_target():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_target_only():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://example.com"])
    assert args.target == "https://example.com"
    assert args.port == 8080
    assert args.verbose is False


def test_parser_all_options():
    parser = _build_parser()
    args = parser.parse_args([
        "--target", "https://server.com",
        "--port", "9090",
        "--verbose",
        "--log-level", "DEBUG",
    ])
    assert args.target == "https://server.com"
    assert args.port == 9090
    assert args.verbose is True
    assert args.log_level == "DEBUG"


def test_parser_file_target():
    parser = _build_parser()
    args = parser.parse_args(["--target", "schemas/mcp_calendar_schemas.json"])
    assert args.target == "schemas/mcp_calendar_schemas.json"


def test_parser_file_uri_target():
    parser = _build_parser()
    args = parser.parse_args(["--target", "file:///abs/path/schemas.json"])
    assert args.target == "file:///abs/path/schemas.json"


def test_parser_stdio_cmd_option():
    """--stdio-cmd doit setter args.target (alias)."""
    parser = _build_parser()
    cmd = "npx @modelcontextprotocol/server-filesystem /tmp"
    args = parser.parse_args(["--stdio-cmd", cmd])
    assert args.target == cmd


def test_parser_stdio_cmd_with_verbose():
    parser = _build_parser()
    args = parser.parse_args([
        "--stdio-cmd", "npx @mcp/server /tmp",
        "--verbose",
    ])
    assert args.target == "npx @mcp/server /tmp"
    assert args.verbose is True


def test_parser_target_and_stdio_cmd_mutually_exclusive():
    """--target et --stdio-cmd ne doivent pas être utilisés ensemble."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "https://x.com", "--stdio-cmd", "npx x"])


def test_parser_mode_defaults_to_stdio():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://x.com"])
    assert args.mode == "stdio"


def test_parser_mode_http():
    parser = _build_parser()
    args = parser.parse_args(["--target", "https://x.com", "--mode", "http", "--port", "9000"])
    assert args.mode == "http"
    assert args.port == 9000


def test_parser_mode_invalid_value():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "x", "--mode", "websocket"])


def test_parser_invalid_log_level():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "x", "--log-level", "INVALID"])


# ─── test que main() s'arrête proprement sur une cible invalide ───────────────

def test_main_exits_on_nonexistent_file(capsys):
    """main() doit gérer l'erreur proprement si le fichier n'existe pas."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--target", "/nonexistent/path/file.json"])
    assert exc_info.value.code != 0


# ─── test que main() fonctionne avec un fichier JSON valide ──────────────────

def test_main_connects_and_serve_called(tmp_path, monkeypatch):
    """Vérifie que main() appelle connect() et serve() sans crash."""
    import json
    import asyncio

    schema_file = tmp_path / "test.json"
    schema_file.write_text(json.dumps([
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]), encoding="utf-8")

    serve_called = []

    # Patch RefractProxy.serve pour éviter de démarrer le vrai serveur stdio
    import refract_proxy as rp

    original_serve = rp.RefractProxy.serve

    async def mock_serve(self):
        serve_called.append(True)

    monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

    main(["--target", str(schema_file), "--verbose"])

    assert serve_called, "serve() doit avoir été appelé"


def test_main_verbose_does_not_write_diagnostics_to_stdout(tmp_path, monkeypatch, capsys):
    """Le mode verbose doit écrire les logs Refract sur stderr, pas stdout."""
    import json

    schema_file = tmp_path / "verbose-fixture.json"
    schema_file.write_text(json.dumps([
        {
            "name": "verbose_tool",
            "description": "A verbose test tool",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]), encoding="utf-8")

    import refract_proxy as rp

    async def mock_serve(self):
        return None

    monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

    main(["--target", str(schema_file), "--verbose"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[Refract]" in captured.err


def test_main_mode_http_calls_serve_http(tmp_path, monkeypatch):
    """--mode http doit appeler serve_http() et NE PAS appeler serve() (stdio)."""
    import json

    schema_file = tmp_path / "test.json"
    schema_file.write_text(json.dumps([
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]), encoding="utf-8")

    serve_http_called = []
    serve_called = []

    import refract_proxy as rp

    async def mock_serve_http(self):
        serve_http_called.append(True)

    async def mock_serve(self):
        serve_called.append(True)

    monkeypatch.setattr(rp.RefractProxy, "serve_http", mock_serve_http)
    monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

    main(["--target", str(schema_file), "--mode", "http", "--port", "9123"])

    assert serve_http_called, "serve_http() doit avoir été appelé en mode http"
    assert not serve_called, "serve() (stdio) ne doit pas être appelé en mode http"


def test_main_mode_stdio_calls_serve_not_serve_http(tmp_path, monkeypatch):
    """--mode stdio (défaut) doit appeler serve() et NE PAS appeler serve_http()."""
    import json

    schema_file = tmp_path / "test.json"
    schema_file.write_text(json.dumps([
        {"name": "t", "description": "d", "inputSchema": {"type": "object", "properties": {}}}
    ]), encoding="utf-8")

    serve_http_called = []
    serve_called = []

    import refract_proxy as rp

    async def mock_serve_http(self):
        serve_http_called.append(True)

    async def mock_serve(self):
        serve_called.append(True)

    monkeypatch.setattr(rp.RefractProxy, "serve_http", mock_serve_http)
    monkeypatch.setattr(rp.RefractProxy, "serve", mock_serve)

    main(["--target", str(schema_file)])  # mode par défaut = stdio

    assert serve_called, "serve() doit avoir été appelé en mode stdio"
    assert not serve_http_called, "serve_http() ne doit pas être appelé en mode stdio"


# ─── test que l'entry point est importable ───────────────────────────────────

def test_main_is_callable():
    from refract_cli import main as cli_main
    assert callable(cli_main)
