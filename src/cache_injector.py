"""
cache_injector — Anthropic prompt caching for MCP tools compressed by Refract.

Anthropic prompt caching reduces the cost of repetitive tokens from $3.00/M
to $0.30/M (10x cheaper) for cache hits.

MCP schemas in a Refract session don't change between requests
-> perfect candidates for caching.

Anthropic rule: set cache_control on the LAST element of the block
you want to cache. Everything before that element is included in the cache.

Anthropic pricing (Claude Sonnet):
  Standard input : $3.00/M tokens
  Cache write    : $3.75/M tokens  (first call — cache population)
  Cache read     : $0.30/M tokens  (subsequent calls — cache hits)
"""

from __future__ import annotations

# ─── Anthropic pricing ───────────────────────────────────────────────────────
PRICE_INPUT_USD_PER_M: float = 3.00    # standard input tokens
PRICE_WRITE_USD_PER_M: float = 3.75   # cache write (first call)
PRICE_READ_USD_PER_M: float = 0.30    # cache read  (subsequent hits)


class CacheInjector:
    """Injects cache_control into MCP tools and calculates Anthropic savings."""

    # ── injection ─────────────────────────────────────────────────────────── #

    @staticmethod
    def inject_cache_control(tools: list[dict]) -> list[dict]:
        """Adds ``cache_control: {type: ephemeral}`` to the last tool in the list.

        This is the Anthropic rule: mark the *last* element of the block
        you want to cache. All preceding elements are automatically
        included in the cache.

        MCP tools don't change during a Refract session -> caching is
        100% effective from the second call onward.

        Args:
            tools: list of MCP tool dicts (Anthropic API or raw MCP format).
                   Each dict contains at minimum ``"name"`` and ``"inputSchema"``.

        Returns:
            New list (shallow copy) with cache_control on the last element.
            Returns the list unchanged if empty.
        """
        if not tools:
            return tools
        result = [dict(t) for t in tools]
        result[-1] = {**result[-1], "cache_control": {"type": "ephemeral"}}
        return result

    # ── savings estimation ────────────────────────────────────────────────── #

    @staticmethod
    def estimate_cache_savings(
        tokens: int,
        requests_per_day: int,
        days: int = 30,
    ) -> dict:
        """Calculates savings from combining Refract + Anthropic prompt caching.

        Compares two scenarios over ``days`` days:

        **Scenario A — no cache, no Refract**: send ``tokens`` tokens on every
        request at the standard input price ($3.00/M).

        **Scenario B — with cache, with Refract**: first call = cache write
        ($3.75/M), subsequent calls = cache read ($0.30/M). Tokens here
        are already compressed by Refract — the function expects the count
        *after* compression.

        Args:
            tokens: number of schema tokens (after Refract compression).
            requests_per_day: number of agent requests per day.
            days: simulation duration (default: 30 days).

        Returns:
            dict with:
            - ``cout_sans_cache_sans_refract`` (float): scenario A cost in USD
            - ``cout_avec_cache_avec_refract`` (float): scenario B cost in USD
            - ``economie_totale_usd`` (float): A - B, always >= 0
            - ``reduction_pct`` (float): relative reduction in %
        """
        total_requests = requests_per_day * days

        # Scenario A: tokens x $3/M x total_requests
        cout_sans = tokens * PRICE_INPUT_USD_PER_M / 1_000_000 * total_requests

        # Scenario B: cache write (1st call) + cache reads (subsequent calls)
        cout_avec = (
            tokens * PRICE_WRITE_USD_PER_M / 1_000_000
            + tokens * PRICE_READ_USD_PER_M / 1_000_000 * max(0, total_requests - 1)
        )

        economie = max(0.0, cout_sans - cout_avec)
        reduction_pct = round(economie / cout_sans * 100, 1) if cout_sans else 0.0

        return {
            "cout_sans_cache_sans_refract": round(cout_sans, 6),
            "cout_avec_cache_avec_refract": round(cout_avec, 6),
            "economie_totale_usd": round(economie, 6),
            "reduction_pct": reduction_pct,
        }

    # ── Anthropic API format helpers ──────────────────────────────────────── #

    @staticmethod
    def to_anthropic_format(tools: list[dict], use_cache: bool = True) -> list[dict]:
        """Converts MCP tools (inputSchema) to Anthropic API format (input_schema).

        The field is called ``inputSchema`` in MCP and ``input_schema`` in the
        Anthropic API. This method renames the field and optionally injects
        ``cache_control`` on the last tool.

        Args:
            tools: list of MCP dicts (``name``, ``description``, ``inputSchema``).
            use_cache: if True, injects cache_control on the last tool.

        Returns:
            List of dicts in Anthropic API format.
        """
        converted = [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", t.get("input_schema", {})),
            }
            for t in tools
        ]
        return CacheInjector.inject_cache_control(converted) if use_cache else converted
