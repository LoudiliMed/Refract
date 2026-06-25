"""
test_install.py — Vérifie l'installeur Claude Desktop (refract_install).

Couvre : création du fichier de config absent, préservation des clés
existantes, overwrite, refract-proxy introuvable, commande custom, et
récupération propre d'un JSON invalide.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import refract_install
from refract_install import add_entry, get_config_path, load_config, save_config


@pytest.fixture
def config_path(tmp_path):
    return tmp_path / "claude_desktop_config.json"


# ─── Test 1: fichier absent → créé avec la bonne structure ────────────────────

def test_missing_config_is_created(config_path):
    assert not config_path.exists()

    config = load_config(config_path)          # absent → structure vide
    assert config == {"mcpServers": {}}

    add_entry(config, "refract-custom", {"command": "p", "args": []})
    save_config(config_path, config)

    assert config_path.exists()
    written = json.loads(config_path.read_text())
    assert "mcpServers" in written
    assert written["mcpServers"]["refract-custom"] == {"command": "p", "args": []}


# ─── Test 2: config existante → entrée ajoutée, préférences préservées ────────

def test_existing_keys_preserved(config_path):
    config_path.write_text(json.dumps({
        "coworkUserFilesPath": "/Users/me/files",
        "preferences": {"theme": "dark"},
        "mcpServers": {"some-other": {"command": "x"}},
    }))

    config = load_config(config_path)
    add_entry(config, "refract-filesystem", {"command": "proxy", "args": ["a"]})
    save_config(config_path, config)

    written = json.loads(config_path.read_text())
    # nouvelle entrée présente
    assert written["mcpServers"]["refract-filesystem"] == {"command": "proxy", "args": ["a"]}
    # tout le reste intact
    assert written["coworkUserFilesPath"] == "/Users/me/files"
    assert written["preferences"] == {"theme": "dark"}
    assert written["mcpServers"]["some-other"] == {"command": "x"}


# ─── Test 3: entrée déjà présente → overwrite ─────────────────────────────────

def test_overwrite_behaviour(config_path):
    config = {"mcpServers": {"refract-filesystem": {"command": "old"}}}

    # sans overwrite → refusé, valeur inchangée
    added = add_entry(config, "refract-filesystem", {"command": "new"}, overwrite=False)
    assert added is False
    assert config["mcpServers"]["refract-filesystem"] == {"command": "old"}

    # avec overwrite → remplacé
    added = add_entry(config, "refract-filesystem", {"command": "new"}, overwrite=True)
    assert added is True
    assert config["mcpServers"]["refract-filesystem"] == {"command": "new"}


# ─── Test 4: refract-proxy introuvable → message clair + exit ─────────────────

def test_proxy_not_found(monkeypatch, capsys, config_path):
    monkeypatch.setattr(refract_install.shutil, "which", lambda _: None)
    monkeypatch.setattr(refract_install, "get_config_path", lambda: config_path)

    with pytest.raises(SystemExit) as exc:
        refract_install.main([])

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "not found" in out
    assert "pip install refract-mcp --break-system-packages" in out


# ─── Test 5: commande custom → enregistrée correctement ───────────────────────

def test_custom_command_saved(monkeypatch, config_path):
    monkeypatch.setattr(refract_install.shutil, "which", lambda _: "/abs/refract-proxy")
    monkeypatch.setattr(refract_install, "get_config_path", lambda: config_path)

    # menu choix 3 + commande custom
    answers = iter(["3", "npx my-server --flag"])
    monkeypatch.setattr(refract_install, "_ask", lambda *a, **k: next(answers))

    refract_install.main([])

    written = json.loads(config_path.read_text())
    entry = written["mcpServers"]["refract-custom"]
    assert entry["command"] == "/abs/refract-proxy"
    assert entry["args"] == ["--stdio-cmd", "npx my-server --flag"]


# ─── Test 6: JSON invalide → sauvegardé en .bak, recréé proprement ────────────

def test_invalid_json_backed_up_and_recreated(config_path):
    config_path.write_text("{ this is not valid json ]")

    config = load_config(config_path)
    assert config == {"mcpServers": {}}

    backup = config_path.with_suffix(config_path.suffix + ".bak")
    assert backup.exists()
    assert backup.read_text() == "{ this is not valid json ]"

    # on peut ensuite écrire une config propre
    add_entry(config, "refract-code", {"command": "srv", "args": ["--root", "."]})
    save_config(config_path, config)
    written = json.loads(config_path.read_text())
    assert written["mcpServers"]["refract-code"]["command"] == "srv"


# ─── Bonus: mode 2 (refract-server-install) ───────────────────────────────────

def test_server_install_filesystem_default(monkeypatch, tmp_path, config_path):
    monkeypatch.setattr(refract_install.shutil, "which", lambda _: "/abs/refract-server")
    monkeypatch.setattr(refract_install, "get_config_path", lambda: config_path)
    monkeypatch.setattr(refract_install, "_ask", lambda *a, **k: str(tmp_path))

    refract_install.main_server([])

    written = json.loads(config_path.read_text())
    entry = written["mcpServers"]["refract-code"]
    assert entry["command"] == "/abs/refract-server"
    assert entry["args"] == ["--root", str(tmp_path)]


def test_get_config_path_platform(monkeypatch):
    monkeypatch.setattr(refract_install.sys, "platform", "darwin")
    p = get_config_path()
    assert p.name == "claude_desktop_config.json"
    assert "Claude" in str(p)
