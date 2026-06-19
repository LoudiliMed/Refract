"""
token_counter — Haiku token cost estimation for Refract context compression.

Haiku 4.5 rates:
  Input  : $1.00/M tokens
  Output : $5.00/M tokens  (assumed 500 default response tokens)
"""
from __future__ import annotations

import ast_extractor as ae

_INPUT_COST_PER_M: float = 1.00    # Haiku input
_OUTPUT_COST_PER_M: float = 5.00   # Haiku output
_OUTPUT_TOKENS_DEFAULT: int = 500   # typical short answer


def compute_stats(tokens_before: int, tokens_after: int) -> dict:
    """Returns cost and savings metrics for a before/after token count pair.

    Pricing model: Haiku 4.5 — $1/M input, $5/M output.
    A constant 500-token output is assumed for every request.

    Args:
        tokens_before: input tokens without compression.
        tokens_after: input tokens after compression.

    Returns:
        dict with keys:
          cost_before_usd, cost_after_usd  – total cost (input + output)
          reduction_pct                    – input token reduction percentage
          savings_usd                      – input-side cost saved
    """
    output_cost = _OUTPUT_TOKENS_DEFAULT * _OUTPUT_COST_PER_M / 1_000_000
    cost_before = tokens_before * _INPUT_COST_PER_M / 1_000_000 + output_cost
    cost_after = tokens_after * _INPUT_COST_PER_M / 1_000_000 + output_cost
    reduction = (
        round((tokens_before - tokens_after) / tokens_before * 100, 6)
        if tokens_before else 0.0
    )
    savings = (tokens_before - tokens_after) * _INPUT_COST_PER_M / 1_000_000
    return {
        "cost_before_usd": cost_before,
        "cost_after_usd": cost_after,
        "reduction_pct": reduction,
        "savings_usd": savings,
    }


def stats_fichier(source: str) -> dict:
    """Compresses ``source`` and returns token counts + reduction percentage.

    Used by llm_client.py to report context compression before sending
    the prompt to the LLM.
    """
    compressed = ae.compress(source)
    tokens_before = ae.count_tokens(source)
    tokens_after = ae.count_tokens(compressed)
    s = compute_stats(tokens_before, tokens_after)
    return {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "reduction_pct": s["reduction_pct"],
    }
