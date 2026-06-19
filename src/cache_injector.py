"""
cache_injector — Prompt caching Anthropic pour les tools MCP compressés par Refract.

Le prompt caching Anthropic réduit le coût des tokens répétitifs de 3,00 $/M
à 0,30 $/M (×10 moins cher) pour les cache hits.

Les schémas MCP d'une session Refract ne changent pas d'une requête à l'autre
→ candidats parfaits au caching.

Règle Anthropic : on pose cache_control sur le DERNIER élément du bloc
qu'on veut cacher. Tout ce qui précède cet élément est inclus dans le cache.

Tarifs Anthropic (Claude Sonnet) :
  Input standard  : 3,00 $/M tokens
  Cache write     : 3,75 $/M tokens  (premier appel — mise en cache)
  Cache read      : 0,30 $/M tokens  (appels suivants — hits)
"""

from __future__ import annotations

# ─── tarifs Anthropic ────────────────────────────────────────────────────────
PRICE_INPUT_USD_PER_M: float = 3.00    # tokens input standard
PRICE_WRITE_USD_PER_M: float = 3.75   # cache write (premier appel)
PRICE_READ_USD_PER_M: float = 0.30    # cache read  (hits suivants)


class CacheInjector:
    """Injecte cache_control dans les tools MCP et calcule les économies Anthropic."""

    # ── injection ─────────────────────────────────────────────────────────── #

    @staticmethod
    def inject_cache_control(tools: list[dict]) -> list[dict]:
        """Ajoute ``cache_control: {type: ephemeral}`` sur le dernier tool de la liste.

        C'est la règle Anthropic : on marque le *dernier* élément du bloc
        qu'on veut mettre en cache. Tous les éléments précédents sont inclus
        automatiquement dans le cache.

        Les tools MCP ne changent pas au fil d'une session Refract → caching
        100 % efficace dès le deuxième appel.

        Args:
            tools: liste de dicts tools MCP (format Anthropic API ou MCP brut).
                   Chaque dict contient au minimum ``"name"`` et ``"inputSchema"``.

        Returns:
            Nouvelle liste (shallow copy) avec cache_control sur le dernier élément.
            Retourne la liste inchangée si elle est vide.
        """
        if not tools:
            return tools
        result = [dict(t) for t in tools]
        result[-1] = {**result[-1], "cache_control": {"type": "ephemeral"}}
        return result

    # ── estimation des économies ──────────────────────────────────────────── #

    @staticmethod
    def estimate_cache_savings(
        tokens: int,
        requests_per_day: int,
        days: int = 30,
    ) -> dict:
        """Calcule les économies de la combinaison Refract + prompt cache Anthropic.

        Compare deux scénarios sur ``days`` jours :

        **Scénario A — sans cache, sans Refract** : on envoie ``tokens`` tokens
        à chaque requête au tarif input standard (3,00 $/M).

        **Scénario B — avec cache, avec Refract** : premier appel = cache write
        (3,75 $/M), appels suivants = cache read (0,30 $/M). Les tokens ici
        sont ceux déjà compressés par Refract — la fonction attend donc le
        compte *après* compression.

        Args:
            tokens: nombre de tokens des schémas (après compression Refract).
            requests_per_day: nombre de requêtes agent par jour.
            days: durée de la simulation (défaut : 30 jours).

        Returns:
            dict avec :
            - ``cout_sans_cache_sans_refract`` (float) : coût scénario A en USD
            - ``cout_avec_cache_avec_refract`` (float) : coût scénario B en USD
            - ``economie_totale_usd`` (float) : A − B, toujours ≥ 0
            - ``reduction_pct`` (float) : réduction relative en %
        """
        total_requests = requests_per_day * days

        # Scénario A : tokens × 3$/M × total_requêtes
        cout_sans = tokens * PRICE_INPUT_USD_PER_M / 1_000_000 * total_requests

        # Scénario B : cache write (1er appel) + cache reads (appels suivants)
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

    # ── helpers format Anthropic API ──────────────────────────────────────── #

    @staticmethod
    def to_anthropic_format(tools: list[dict], use_cache: bool = True) -> list[dict]:
        """Convertit des tools MCP (inputSchema) en format Anthropic API (input_schema).

        Le champ s'appelle ``inputSchema`` en MCP et ``input_schema`` dans l'API
        Anthropic. Cette méthode renomme le champ et injecte optionnellement
        ``cache_control`` sur le dernier tool.

        Args:
            tools: liste de dicts MCP (``name``, ``description``, ``inputSchema``).
            use_cache: si True, injecte cache_control sur le dernier tool.

        Returns:
            Liste de dicts au format Anthropic API.
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
