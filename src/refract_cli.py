"""
refract_cli — CLI entry point for the Refract proxy.

Usage:
    # stdio mode (default) — local subprocess, Claude Desktop / Cursor integration
    refract-proxy --target "npx @modelcontextprotocol/server-filesystem /tmp" --mode stdio
    refract-proxy --stdio-cmd "npx @modelcontextprotocol/server-filesystem /tmp"

    # http mode — agent connects via a network URL (http://localhost:8080/sse)
    refract-proxy --target "https://my-mcp-server.com" --mode http --port 8080

    # JSON schema file (tests / static catalogue), both modes work
    refract-proxy --target schemas/mcp_calendar_schemas.json --mode http --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="refract-proxy",
        description=(
            "Refract — MCP proxy that compresses tool schemas.\n"
            "Plug it between your agent (Claude, Cursor…) and any MCP server.\n\n"
            "Examples:\n"
            "  refract-proxy --target https://my-server.com --verbose\n"
            "  refract-proxy --stdio-cmd \"npx @modelcontextprotocol/server-filesystem /tmp\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --target and --stdio-cmd are two ways to specify the target (mutually exclusive)
    target_group = p.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        dest="target",
        metavar="URL",
        help=(
            "MCP target: HTTP/SSE URL (https://…), JSON file path, "
            "or stdio command (\"npx @mcp/server /path\")"
        ),
    )
    target_group.add_argument(
        "--stdio-cmd",
        dest="target",
        metavar="CMD",
        help=(
            "Alias for --target for stdio commands. "
            "Ex: \"npx @modelcontextprotocol/server-filesystem /tmp\""
        ),
    )

    p.add_argument(
        "--port",
        type=int,
        default=8080,
        metavar="PORT",
        help="Local proxy port in HTTP/SSE mode (default: 8080)",
    )
    p.add_argument(
        "--mode",
        choices=["stdio", "http"],
        default="stdio",
        help=(
            "Proxy transport: stdio (default — local subprocess, "
            "Claude Desktop/Cursor) or http (SSE network, URL http://localhost:<port>/sse)"
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print token counts before/after for each request",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: WARNING)",
    )
    return p


async def _run(args: argparse.Namespace) -> None:
    from refract_proxy import RefractProxy

    proxy = RefractProxy(
        target_url=args.target,
        port=args.port,
        verbose=args.verbose,
    )

    print(f"[Refract] Connecting to {args.target} …", file=sys.stderr)
    await proxy.connect()

    n = len(proxy._tools)
    if args.mode == "http":
        print(f"[Refract] {n} tools loaded. Starting MCP proxy (HTTP/SSE) …", file=sys.stderr)
        await proxy.serve_http()
    else:
        print(f"[Refract] {n} tools loaded. Starting MCP proxy (stdio) …", file=sys.stderr)
        print("[Refract] Ready — configure your agent to use this proxy as an MCP server.", file=sys.stderr)
        await proxy.serve()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[Refract] Proxy stopped.", file=sys.stderr)
    except Exception as exc:
        print(f"[Refract] Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
