"""
refract_cli — Point d'entrée CLI du proxy Refract.ai.

Usage :
    # Mode stdio (défaut) — sous-processus local, intégration Claude Desktop / Cursor
    refract-proxy --target "npx @modelcontextprotocol/server-filesystem /tmp" --mode stdio
    refract-proxy --stdio-cmd "npx @modelcontextprotocol/server-filesystem /tmp"

    # Mode http — l'agent se connecte via une URL réseau (http://localhost:8080/sse)
    refract-proxy --target "https://mon-serveur-mcp.com" --mode http --port 8080

    # Fichier JSON de schémas (tests / catalogue statique), les deux modes marchent
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
            "Refract.ai — Proxy MCP qui compresse les schémas de tools.\n"
            "Branchez-le entre votre agent (Claude, Cursor…) et n'importe quel serveur MCP.\n\n"
            "Exemples :\n"
            "  refract-proxy --target https://mon-serveur.com --verbose\n"
            "  refract-proxy --stdio-cmd \"npx @modelcontextprotocol/server-filesystem /tmp\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --target et --stdio-cmd sont deux façons d'indiquer la cible (mutuellement exclusifs)
    target_group = p.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        dest="target",
        metavar="URL",
        help=(
            "Cible MCP : URL HTTP/SSE (https://…), chemin de fichier JSON, "
            "ou commande stdio (\"npx @mcp/server /path\")"
        ),
    )
    target_group.add_argument(
        "--stdio-cmd",
        dest="target",
        metavar="CMD",
        help=(
            "Alias de --target pour les commandes stdio. "
            "Ex : \"npx @modelcontextprotocol/server-filesystem /tmp\""
        ),
    )

    p.add_argument(
        "--port",
        type=int,
        default=8080,
        metavar="PORT",
        help="Port local du proxy en mode HTTP/SSE (défaut : 8080)",
    )
    p.add_argument(
        "--mode",
        choices=["stdio", "http"],
        default="stdio",
        help=(
            "Transport du proxy : stdio (défaut — sous-processus local, "
            "Claude Desktop/Cursor) ou http (SSE réseau, URL http://localhost:<port>/sse)"
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche les tokens avant/après pour chaque requête",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de log (défaut : WARNING)",
    )
    return p


async def _run(args: argparse.Namespace) -> None:
    from refract_proxy import RefractProxy

    proxy = RefractProxy(
        target_url=args.target,
        port=args.port,
        verbose=args.verbose,
    )

    print(f"[Refract] Connexion à {args.target} …")
    await proxy.connect()

    n = len(proxy._tools)
    if args.mode == "http":
        print(f"[Refract] {n} tools chargés. Démarrage du proxy MCP (HTTP/SSE) …")
        await proxy.serve_http()
    else:
        print(f"[Refract] {n} tools chargés. Démarrage du proxy MCP (stdio) …")
        print("[Refract] Prêt — configurez votre agent pour utiliser ce proxy comme serveur MCP.")
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
        print("\n[Refract] Arrêt du proxy.")
    except Exception as exc:
        print(f"[Refract] Erreur fatale : {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
