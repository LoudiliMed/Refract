"""
refract_install — one-shot installers that wire Refract into Claude Desktop.

Two CLI entry points:

    refract-install         # wrap an MCP server with the refract-proxy (Mode 1)
    refract-server-install  # expose your codebase via refract-server (Mode 2)

Both edit Claude Desktop's ``claude_desktop_config.json`` SAFELY: the file is
loaded whole, a ``.bak`` backup is taken, only the new ``mcpServers`` entry is
added, and the original is restored automatically if the write fails. No other
keys (preferences, coworkUserFilesPath, existing servers…) are ever touched.

The interactive prompts live in ``main`` / ``main_server``; all the file logic
is in small pure helpers so it can be unit-tested without stdin/stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# ─── config location ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the Claude Desktop config path for the current platform."""
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    # linux / other
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


# ─── load / save ─────────────────────────────────────────────────────────────


def load_config(path: Path) -> dict:
    """Load the config JSON.

    - Missing file        → returns ``{"mcpServers": {}}`` (nothing written yet).
    - Invalid JSON        → backs the corrupt file up to ``.bak`` first, then
                            returns a fresh ``{"mcpServers": {}}``.
    - Valid JSON          → returned as-is, with ``mcpServers`` ensured present.
    """
    if not path.exists():
        return {"mcpServers": {}}

    raw = path.read_text(encoding="utf-8")
    try:
        config = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"  ! Existing config was invalid JSON. Backed up to: {backup}")
        config = {}

    if not isinstance(config, dict):
        config = {}
    config.setdefault("mcpServers", {})
    return config


def save_config(path: Path, config: dict) -> None:
    """Write ``config`` to ``path`` safely.

    Backs the original file up to ``.bak`` first, then writes with indent=2.
    If the write raises, the backup is restored and the error re-raised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    backup: Path | None = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

    try:
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    except Exception:
        if backup and backup.exists():
            shutil.copy2(backup, path)
            print("  ! Write failed — restored original config from backup.")
        raise


# ─── entry management ────────────────────────────────────────────────────────


def add_entry(
    config: dict,
    name: str,
    entry: dict,
    *,
    overwrite: bool = False,
) -> bool:
    """Add ``entry`` under ``config['mcpServers'][name]``.

    Returns ``True`` if the entry was added/updated, ``False`` if it already
    existed and ``overwrite`` was not granted. Only the single key is touched.
    """
    servers = config.setdefault("mcpServers", {})
    if name in servers and not overwrite:
        return False
    servers[name] = entry
    return True


# ─── tool discovery ──────────────────────────────────────────────────────────


def _which_or_die(executable: str) -> str:
    """Return the absolute path of ``executable`` or exit with guidance."""
    path = shutil.which(executable)
    if not path:
        print(f"Error: '{executable}' not found on your PATH.")
        print("It ships with the refract-mcp package. Install it with:")
        print("    pip install refract-mcp --break-system-packages")
        sys.exit(1)
    return path


# ─── small prompt helpers ────────────────────────────────────────────────────


def _ask(prompt: str, default: str = "") -> str:
    """input() wrapper that returns ``default`` on empty answer / EOF."""
    try:
        answer = input(prompt).strip()
    except EOFError:
        return default
    return answer or default


def _confirm(prompt: str) -> bool:
    """Yes/No prompt, defaults to No."""
    return _ask(prompt).lower() in ("y", "yes")


# ─── Mode 1: wrap an MCP server with the proxy ─────────────────────────────────

_MENU = """
Which MCP server do you want to wrap with Refract?

  1. Filesystem (npx @modelcontextprotocol/server-filesystem)
  2. Sequential Thinking (npx @modelcontextprotocol/server-sequential-thinking)
  3. Custom (I will type my own command)
"""


def _resolve_proxy_choice() -> tuple[str, list[str]]:
    """Interactive menu → (server_name, target_args).

    ``target_args`` is the command that the proxy will wrap, already split into
    argv form (passed after ``--stdio-cmd``).
    """
    print(_MENU)
    choice = _ask("Choose [1/2/3]: ", "1")

    if choice == "1":
        home = str(Path.home())
        folder = _ask(
            f"\nWhich folder should the filesystem server access?\n"
            f"(default: your home directory {home}): ",
            home,
        )
        target = ["npx", "@modelcontextprotocol/server-filesystem", folder]
        return "refract-filesystem", target

    if choice == "2":
        target = ["npx", "@modelcontextprotocol/server-sequential-thinking"]
        return "refract-sequential-thinking", target

    if choice == "3":
        raw = _ask("\nEnter the full MCP server command: ")
        if not raw:
            print("Error: empty command.")
            sys.exit(1)
        return "refract-custom", raw.split()

    print(f"Error: invalid choice '{choice}'.")
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    config_path = get_config_path()
    proxy_path = _which_or_die("refract-proxy")

    name, target_args = _resolve_proxy_choice()
    target_cmd = " ".join(target_args)

    config = load_config(config_path)

    overwrite = True
    if name in config.get("mcpServers", {}):
        overwrite = _confirm(f"\n{name} already exists. Overwrite? [y/N]: ")
        if not overwrite:
            print("Aborted — nothing changed.")
            return

    entry = {"command": proxy_path, "args": ["--stdio-cmd", target_cmd]}
    add_entry(config, name, entry, overwrite=overwrite)
    save_config(config_path, config)

    print(f"\nAdded {name} to Claude Desktop config.")
    print(f"  Path:   {proxy_path}")
    print(f"  Target: {target_cmd}")
    print("\nRestart Claude Desktop to activate.")
    if name == "refract-filesystem":
        print('Then ask Claude: "List the files in my home directory"')

    print(
        "\nNote: some MCP servers require additional authentication\n"
        "(Google OAuth, GitHub token, Slack API key, etc.).\n"
        "If your server needs credentials, follow its own setup\n"
        "instructions before restarting Claude Desktop.\n"
        "\n"
        "Refract itself needs no credentials — it only compresses\n"
        "what passes through."
    )


# ─── Mode 2: expose codebase via refract-server ───────────────────────────────


def _resolve_server_choice() -> str:
    """Ask which folder refract-server should analyze. Returns absolute path."""
    folder = _ask(
        "Which folder contains the code you want Claude to analyze?\n"
        "(default: current directory .): ",
        ".",
    )
    return str(Path(folder).expanduser().resolve())


def main_server(argv: list[str] | None = None) -> None:
    config_path = get_config_path()
    server_path = _which_or_die("refract-server")

    root = _resolve_server_choice()
    name = "refract-code"

    config = load_config(config_path)

    overwrite = True
    if name in config.get("mcpServers", {}):
        overwrite = _confirm(f"\n{name} already exists. Overwrite? [y/N]: ")
        if not overwrite:
            print("Aborted — nothing changed.")
            return

    entry = {"command": server_path, "args": ["--root", root]}
    add_entry(config, name, entry, overwrite=overwrite)
    save_config(config_path, config)

    print(f"\nAdded {name} to Claude Desktop config.")
    print(f"  Path: {server_path}")
    print(f"  Root: {root}")
    print("\nRestart Claude Desktop, then ask Claude:")
    print('  "Which functions break if I change authenticate()?"')
    print('  "Find all security risks in my codebase"')
    print('  "Show me the blast radius of the compress function"')


if __name__ == "__main__":
    main()
