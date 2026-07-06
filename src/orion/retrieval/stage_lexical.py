"""Stage 3: BM25 lexical retrieval and Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_bm25_available = False
try:
    from rank_bm25 import BM25Okapi

    _bm25_available = True
except ImportError:
    pass

# Telecom tokens to preserve intact
_PRESERVE_TOKENS = {"5g", "urllc", "vnf", "embb", "5qi", "v2x", "mec", "ran", "nf"}


def telecom_tokenize(text: str) -> list[str]:
    """Telecom-aware tokenizer: lowercase, split on whitespace+punctuation,
    preserve domain acronyms and numeric identifiers."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_]*", text)
    return tokens


class LexicalRetriever:
    """BM25-based lexical retriever with telecom-aware tokenization."""

    def __init__(self) -> None:
        self._bm25: object | None = None
        self._corpus_tokens: list[list[str]] = []

    def build_index(self, documents: list[str]) -> None:
        """Build BM25 index from document texts.

        Hard-fails if rank_bm25 is missing. A silent BM25 fallback degrades M^B
        retrieval to recency-only (content-blind) without any error — the
        project's recurring "component present but silently not firing" failure
        mode. If BM25 is genuinely not wanted, set enable_bm25=False explicitly
        so the intent is in the config, not an absent dependency.
        """
        if not _bm25_available:
            raise RuntimeError(
                "rank_bm25 is not installed but BM25 retrieval is enabled — this "
                "would silently degrade M^B retrieval to recency-only. Install the "
                "retrieval extra: `uv pip install -e '.[retrieval]'` (or "
                "`pip install rank-bm25`). To run without BM25 on purpose, set "
                "RetrievalConfig(enable_bm25=False)."
            )

        self._corpus_tokens = [telecom_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def query(
        self,
        query_text: str,
        candidate_indices: list[int] | None = None,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Return (index, score) pairs ranked by BM25 score.

        Args:
            query_text: Raw query string.
            candidate_indices: If provided, only score these indices.
            top_k: Number of results to return.

        Returns:
            List of (index, bm25_score) sorted descending.
        """
        if self._bm25 is None:
            return []

        tokens = telecom_tokenize(query_text)
        scores = self._bm25.get_scores(tokens)

        if candidate_indices is not None:
            indexed_scores = [(i, float(scores[i])) for i in candidate_indices]
        else:
            indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]

        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


def rrf_fuse(
    rankings: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple rankings.

    Args:
        rankings: List of ranked lists, each containing (index, score) tuples.
        k: RRF smoothing constant (default 60).

    Returns:
        Fused ranking as list of (index, rrf_score) sorted descending.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
