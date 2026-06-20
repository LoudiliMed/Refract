"""
semantic_router — Embedding-based semantic tool routing for Refract.

Replaces keyword overlap scoring with cosine similarity over
BAAI/bge-small-en-v1.5 embeddings (fastembed, fully local, zero LLM calls).

The model (~130 MB) is downloaded once and cached by fastembed.
At runtime, only ONNX inference is used — no GPU required.

Fallback: if fastembed is not installed, _keyword_score() is used instead.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _keyword_score(query: str, tools: list[dict]) -> str:
    """Keyword overlap fallback — original logic, kept here for reuse."""
    words = set(query.lower().split())
    scores: dict[str, int] = {}
    for t in tools:
        name = t.get("name", "").lower().replace("_", " ")
        desc = t.get("description", "").lower()
        scores[t.get("name", "")] = sum(1 for w in words if w in name or w in desc)
    return max(scores, key=scores.get) if scores else ""


class SemanticRouter:
    """Embedding-based MCP tool selector.

    Zero LLM calls.  Fully deterministic after the model is loaded.
    The embedding model is cached at the class level so it is loaded
    at most once per process regardless of how many instances are created.

    Usage::

        router = SemanticRouter()
        router.index_tools(tools)          # embed all tool descriptions
        name = router.find_best_tool_with_threshold(query, min_score=0.3)
        if name is None:
            ...  # fall back to sending the compact index
    """

    # Class-level model cache: model_name -> TextEmbedding instance
    _model_cache: dict[str, Any] = {}

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._tool_names: list[str] = []
        self._tool_embeddings: np.ndarray | None = None

    # ── model loading ───────────────────────────────────────────────── #

    def _get_model(self):
        """Lazy-load the embedding model (cached at class level)."""
        if self._model_name not in SemanticRouter._model_cache:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is not installed. Run: pip install fastembed"
                ) from exc
            logger.debug("SemanticRouter: loading model '%s' (first use)", self._model_name)
            SemanticRouter._model_cache[self._model_name] = TextEmbedding(self._model_name)
        return SemanticRouter._model_cache[self._model_name]

    # ── indexing ────────────────────────────────────────────────────── #

    def index_tools(self, tools: list[dict]) -> None:
        """Embed all tool descriptions and build the in-memory search index.

        Each tool is represented as the concatenation of its name and
        description — the same text that an agent reads when deciding
        which tool to call.

        Args:
            tools: list of MCP tool dicts with at least ``name`` and
                   optionally ``description``.
        """
        if not tools:
            self._tool_names = []
            self._tool_embeddings = None
            return

        self._tool_names = [t["name"] for t in tools]
        texts = [
            f"{t['name']}: {t.get('description', '')}"
            for t in tools
        ]
        model = self._get_model()
        # fastembed.embed() returns a generator; materialise it into an ndarray
        self._tool_embeddings = np.array(list(model.embed(texts)), dtype=np.float32)
        logger.debug("SemanticRouter: indexed %d tools", len(tools))

    # ── querying ────────────────────────────────────────────────────── #

    def find_best_tool(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        """Embed ``query`` and return the top-k most similar tool names.

        bge-small embeddings are L2-normalised, so the dot product of two
        embedding vectors equals their cosine similarity.

        Args:
            query: natural language query string.
            top_k: number of results to return (default 1).

        Returns:
            List of ``(tool_name, cosine_similarity)`` tuples sorted
            descending by score.  Empty list if no tools are indexed.
        """
        if self._tool_embeddings is None or not self._tool_names:
            return []

        model = self._get_model()
        q_vec = np.array(list(model.embed([query]))[0], dtype=np.float32)

        # dot product == cosine similarity for unit-norm vectors
        scores: np.ndarray = self._tool_embeddings @ q_vec
        k = min(top_k, len(self._tool_names))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._tool_names[int(i)], float(scores[i])) for i in top_idx]

    def find_best_tool_with_threshold(
        self,
        query: str,
        min_score: float = 0.3,
    ) -> str | None:
        """Return the best matching tool only if similarity exceeds *min_score*.

        A query below the threshold is considered ambiguous.  The caller
        should then send the compact index to the agent instead of guessing
        the wrong tool.

        Args:
            query: natural language query.
            min_score: minimum cosine similarity (0–1).  Empirically 0.3
                       works well for bge-small; raise it to 0.5 for
                       stricter routing.

        Returns:
            Tool name string, or ``None`` if confidence is too low.
        """
        results = self.find_best_tool(query, top_k=1)
        if not results:
            return None
        name, score = results[0]
        if score < min_score:
            logger.debug(
                "SemanticRouter: best='%s' score=%.3f < threshold=%.3f — no match",
                name, score, min_score,
            )
            return None
        return name
