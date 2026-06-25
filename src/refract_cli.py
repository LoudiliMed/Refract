"""
refract_cli — CLI entry point for the Refract proxy.

Usage:
    # stdio mode (default) — local subprocess, Claude Desktop / Cursor integration
    refract-proxy --target "npx @modelcontextprotocol/server-filesystem /tmp" --mode stdio
    refract-proxy --stdio-cmd "npx @modelcontextprotocol/server-filesystem /tmp"

    # SSE transport: proxy connects to a remote HTTP/SSE MCP server
    refract-proxy --url https://my-mcp-server.com/sse --mode stdio
    refract-proxy --target https://my-mcp-server.com/sse --transport sse

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

    # --target, --stdio-cmd, and --url specify the target (mutually exclusive)
    target_group = p.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        dest="target",
        metavar="URL",
        help=(
            "MCP target: HTTP/SSE URL (https://…), JSON file path, "
            "or stdio command (\"npx @mcp/server /path\"). "
            "Transport is auto-detected; use --transport to override."
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
    target_group.add_argument(
        "--url",
        dest="url",
        metavar="URL",
        help=(
            "Remote SSE/HTTP MCP endpoint — implies --transport sse. "
            "Ex: https://my-mcp-server.com/sse"
        ),
    )

    p.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=None,
        metavar="{stdio,sse}",
        help=(
            "Transport used to connect to the target MCP server. "
            "Default: auto-detected (http:// URL → sse, command → stdio). "
            "--url always implies sse."
        ),
    )
    p.add_argument(
        "--sse-timeout",
        dest="sse_timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Timeout in seconds for SSE connections (default: 30)",
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
            "How the proxy serves the agent: stdio (default — Claude Desktop/Cursor) "
            "or http (SSE network, URL http://localhost:<port>/sse)"
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

    # Resolve target URL and transport from the mutually exclusive group
    url_flag = getattr(args, "url", None)
    target = args.target if args.target else url_flag
    transport = args.transport or ("sse" if url_flag else None)

    proxy = RefractProxy(
        target_url=target,
        port=args.port,
        verbose=args.verbose,
        transport=transport,
        sse_timeout=args.sse_timeout,
    )

    print(f"[Refract] Connecting to {target} …", file=sys.stderr)
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

    # --url implies SSE; reject contradictory --transport stdio
    url_flag = getattr(args, "url", None)
    if url_flag and args.transport == "stdio":
        parser.error(
            "--url is for SSE endpoints and cannot be combined with --transport stdio. "
            "Use --target instead of --url if you want stdio transport."
        )

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
