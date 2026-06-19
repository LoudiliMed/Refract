"""
test_proxy_mount.py — Vérifie que main.py monte bien le proxy MCP en HTTP/SSE
sur /proxy/sse + /proxy/messages, et que le CORS s'applique aux deux.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _load_main_module(name: str = "main_mod_proxy_mount"):
    main_path = pathlib.Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location(name, main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_mounts_proxy_route():
    mod = _load_main_module()
    mount_paths = [getattr(r, "path", None) for r in mod.app.routes]
    assert "/proxy" in mount_paths


def test_main_proxy_target_default_is_calendar_schemas():
    mod = _load_main_module("main_mod_proxy_mount2")
    assert mod._PROXY_TARGET.endswith("mcp_calendar_schemas.json")


def test_main_proxy_target_overridable_by_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text("[]")
    monkeypatch.setenv("REFRACT_PROXY_TARGET", str(custom))
    mod = _load_main_module("main_mod_proxy_mount3")
    assert mod._PROXY_TARGET == str(custom)


def test_main_refract_proxy_instance_exists():
    mod = _load_main_module("main_mod_proxy_mount4")
    from refract_proxy import RefractProxy
    assert isinstance(mod._refract_proxy, RefractProxy)


def test_health_endpoint_unaffected_by_proxy_mount():
    mod = _load_main_module("main_mod_proxy_mount5")
    result = mod.health()
    assert result["status"] == "ok"


def test_cors_middleware_wraps_whole_app_including_proxy():
    """Le CORSMiddleware doit envelopper toute l'app, /proxy/* en hérite sans config dédiée."""
    mod = _load_main_module("main_mod_proxy_mount6")
    from fastapi.middleware.cors import CORSMiddleware

    middleware_classes = [m.cls for m in mod.app.user_middleware]
    assert CORSMiddleware in middleware_classes


def test_proxy_mount_sse_route_resolves_without_404():
    """/proxy/sse doit être routé vers le handler SSE du sous-app monté.

    Le flux SSE reste ouvert indéfiniment côté serveur (poignée de main MCP) :
    on ne fait donc pas de requête live (risque de bloquer le test), on
    vérifie la résolution de route par introspection statique du Mount,
    comme test_proxy.py::test_build_asgi_app_has_sse_and_messages_routes
    mais à travers le montage réel de main.py.
    """
    mod = _load_main_module("main_mod_proxy_mount7")
    from starlette.routing import Mount

    proxy_mount = next(r for r in mod.app.routes if isinstance(r, Mount) and r.path == "/proxy")
    sub_paths = [getattr(r, "path", None) for r in proxy_mount.app.routes]
    assert "/sse" in sub_paths
    assert any(p and "messages" in p for p in sub_paths)
