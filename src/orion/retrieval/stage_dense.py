"""Stage 2: Dense retrieval via embedding similarity."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_faiss_available = False
try:
    import faiss

    _faiss_available = True
except ImportError:
    pass


class DenseRetriever:
    """Dense retrieval using normalized embeddings and inner-product similarity."""

    def __init__(self) -> None:
        self._index: object | None = None
        self._embeddings: np.ndarray | None = None
        self._use_faiss: bool = False

    def build_index(self, embeddings: list[list[float]]) -> None:
        """Build an index from L2-normalized embeddings."""
        if not embeddings:
            self._embeddings = None
            self._index = None
            return

        arr = np.array(embeddings, dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr = arr / norms
        self._embeddings = arr

        if _faiss_available and len(arr) >= 200:
            dim = arr.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(arr)
            self._use_faiss = True
        else:
            self._use_faiss = False

    def query(
        self,
        query_embedding: list[float],
        candidate_indices: list[int] | None = None,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Return (index, score) pairs ranked by similarity.

        Args:
            query_embedding: L2-normalized query vector.
            candidate_indices: If provided, only search within these indices.
            top_k: Number of results to return.

        Returns:
            List of (original_index, similarity_score) sorted descending.
        """
        if self._embeddings is None:
            return []

        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if candidate_indices is not None:
            # Subset search — always numpy
            subset = self._embeddings[candidate_indices]
            scores = (subset @ q.T).flatten()
            k = min(top_k, len(scores))
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return [(candidate_indices[i], float(scores[i])) for i in top_idx]

        if self._use_faiss and self._index is not None:
            k = min(top_k, self._embeddings.shape[0])
            scores, indices = self._index.search(q, k)
            return [(int(idx), float(s)) for idx, s in zip(indices[0], scores[0]) if idx >= 0]

        # Numpy fallback
        scores = (self._embeddings @ q.T).flatten()
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in top_idx]
