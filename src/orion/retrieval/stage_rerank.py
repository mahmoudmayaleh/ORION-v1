"""Stage 4: Cross-encoder reranking."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_st_available = False
try:
    from sentence_transformers import CrossEncoder

    _st_available = True
except ImportError:
    pass


class CrossEncoderReranker:
    """Cross-encoder reranker for final-stage precision improvement."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model on first use."""
        if not _st_available:
            logger.warning(
                "sentence_transformers not installed — Stage 4 (rerank) disabled. "
                "Install with: pip install sentence-transformers"
            )
            return

        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        indices: list[int],
    ) -> list[tuple[int, float]]:
        """Rerank documents by cross-encoder relevance to query.

        Args:
            query: Query text.
            documents: Document texts corresponding to indices.
            indices: Original corpus indices for each document.

        Returns:
            List of (index, reranker_score) sorted descending.
        """
        if self._model is None:
            self._load_model()

        if self._model is None:
            # Library unavailable — pass through in original order
            return [(idx, 0.0) for idx in indices]

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        results = [(idx, float(s)) for idx, s in zip(indices, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
