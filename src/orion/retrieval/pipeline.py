"""Orchestrator for the 4-stage hybrid retrieval pipeline."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Callable

from .stage_dense import DenseRetriever
from .stage_filter import apply_metadata_filter
from .stage_lexical import LexicalRetriever, rrf_fuse
from .stage_rerank import CrossEncoderReranker
from .types import (
    MemoryEntry,
    RetrievalConfig,
    RetrievalMode,
    RetrievalQuery,
    RetrievalTrace,
    ScoredEntry,
)

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Composes all retrieval stages into a configurable pipeline.

    Dispatches per config.mode:
      - full: filter -> dense -> recency -> RRF(dense, BM25) -> rerank -> top_k
      - no_rerank: filter -> dense -> recency -> RRF(dense, BM25) -> top_k
      - dense_only: filter -> dense -> recency -> top_k
      - keyword_only: filter -> BM25 -> top_k
      - cosine_only: dense over entire corpus -> top_k (skip filter, BM25, rerank)
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.config = config
        self._embed_fn = embed_fn
        self._entries: list[MemoryEntry] = []
        self._entry_index: dict[int, int] = {}
        self._dense = DenseRetriever()
        self._lexical = LexicalRetriever()
        self._reranker: CrossEncoderReranker | None = None

        if config.enable_rerank and config.mode == RetrievalMode.FULL:
            self._reranker = CrossEncoderReranker(config.reranker_model)

    def build(self, entries: list[MemoryEntry]) -> None:
        """Build all indices from a list of entries."""
        self._entries = list(entries)
        self._entry_index = {id(e): i for i, e in enumerate(self._entries)}
        if not entries:
            return

        # Embed entries that lack embeddings
        needs_embedding = [
            i for i, e in enumerate(entries) if e.embedding is None
        ]
        if needs_embedding and self._embed_fn is not None:
            texts = [entries[i].topic + " " + entries[i].content for i in needs_embedding]
            embeddings = self._embed_fn(texts)
            for idx, emb in zip(needs_embedding, embeddings):
                self._entries[idx].embedding = emb

        # Build dense index
        all_embeddings = [e.embedding for e in self._entries if e.embedding is not None]
        if all_embeddings:
            self._dense.build_index(all_embeddings)

        # Build lexical index
        if self.config.enable_bm25:
            documents = [e.topic + " " + e.content for e in self._entries]
            self._lexical.build_index(documents)

    def retrieve(
        self, query: RetrievalQuery
    ) -> tuple[list[ScoredEntry], RetrievalTrace | None]:
        """Execute retrieval pipeline according to config.mode.

        Returns:
            Tuple of (scored_entries, trace). Trace is None unless config.return_trace=True.
        """
        trace = RetrievalTrace() if self.config.return_trace else None
        mode = self.config.mode
        t0 = time.perf_counter()

        if mode == RetrievalMode.COSINE_ONLY:
            return self._cosine_only(query, trace)
        if mode == RetrievalMode.KEYWORD_ONLY:
            return self._keyword_only(query, trace)

        # Stage 1: Metadata filter
        candidates = apply_metadata_filter(self._entries, query)
        candidate_indices = [
            self._entry_index[id(c)] for c in candidates
        ]
        if trace:
            trace.candidates_after_filter = len(candidates)

        if not candidate_indices:
            return [], trace

        # Stage 2: Dense retrieval
        query_emb = self._get_query_embedding(query.text)
        if query_emb is None:
            # No embedding available — fall back to all candidates
            dense_results = [(i, 1.0) for i in candidate_indices]
        else:
            dense_results = self._dense.query(
                query_emb,
                candidate_indices=candidate_indices,
                top_k=self.config.k_after_dense,
            )

        # Stage 2.5: Recency weighting
        if self.config.apply_recency and self.config.recency_tau > 0:
            dense_results = self._apply_recency(dense_results)

        if trace:
            trace.candidates_after_dense = len(dense_results)

        if mode == RetrievalMode.DENSE_ONLY:
            scored = self._build_scored(dense_results, "dense")
            logger.debug("Pipeline (dense_only) took %.3fs", time.perf_counter() - t0)
            return scored[: self.config.k_final], trace

        # Stage 3: BM25 + RRF
        if self.config.enable_bm25:
            bm25_indices = [idx for idx, _ in dense_results]
            bm25_results = self._lexical.query(
                query.text,
                candidate_indices=bm25_indices,
                top_k=self.config.k_after_dense,
            )
            fused = rrf_fuse([dense_results, bm25_results], k=self.config.rrf_k)
        else:
            fused = dense_results
            bm25_results = []

        fused = fused[: self.config.k_after_rrf]
        if trace:
            trace.candidates_after_rrf = len(fused)

        if mode == RetrievalMode.NO_RERANK:
            scored = self._build_scored_with_bm25(fused, dense_results, bm25_results)
            logger.debug("Pipeline (no_rerank) took %.3fs", time.perf_counter() - t0)
            return scored[: self.config.k_final], trace

        # Stage 4: Cross-encoder rerank
        if self._reranker is not None:
            rerank_indices = [idx for idx, _ in fused]
            rerank_docs = [
                self._entries[idx].topic + " " + self._entries[idx].content
                for idx in rerank_indices
            ]
            reranked = self._reranker.rerank(query.text, rerank_docs, rerank_indices)
        else:
            reranked = fused

        if trace:
            trace.candidates_after_rerank = len(reranked)

        scored = self._build_scored_full(reranked, dense_results, bm25_results)
        logger.debug("Pipeline (full) took %.3fs", time.perf_counter() - t0)
        return scored[: self.config.k_final], trace

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a single entry and rebuild indices."""
        self._entries.append(entry)
        if entry.embedding is None and self._embed_fn is not None:
            embeddings = self._embed_fn([entry.topic + " " + entry.content])
            entry.embedding = embeddings[0]
        self._rebuild_indices()

    def remove_entry(self, entry_id: str) -> None:
        """Remove an entry by ID and rebuild indices."""
        self._entries = [e for e in self._entries if e.entry_id != entry_id]
        self._rebuild_indices()

    def update_access(self, entry_id: str) -> None:
        """Update access metadata for an entry."""
        for e in self._entries:
            if e.entry_id == entry_id:
                e.last_accessed_at = datetime.now()
                e.access_count += 1
                break

    def _rebuild_indices(self) -> None:
        """Rebuild all indices from current entries."""
        self._entry_index = {id(e): i for i, e in enumerate(self._entries)}
        all_embeddings = [e.embedding for e in self._entries if e.embedding is not None]
        if all_embeddings:
            self._dense.build_index(all_embeddings)
        if self.config.enable_bm25:
            documents = [e.topic + " " + e.content for e in self._entries]
            self._lexical.build_index(documents)

    def _get_query_embedding(self, text: str) -> list[float] | None:
        """Get embedding for query text."""
        if self._embed_fn is None:
            return None
        result = self._embed_fn([text])
        return result[0] if result else None

    def _apply_recency(
        self, results: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """Apply exponential recency decay to dense scores."""
        now = datetime.now()
        tau = self.config.recency_tau
        weighted = []
        for idx, score in results:
            entry = self._entries[idx]
            delta_days = (now - entry.last_accessed_at).total_seconds() / 86400.0
            weight = math.exp(-tau * delta_days)
            weighted.append((idx, score * weight))
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted

    def _cosine_only(
        self, query: RetrievalQuery, trace: RetrievalTrace | None
    ) -> tuple[list[ScoredEntry], RetrievalTrace | None]:
        """Pure dense retrieval over entire corpus, no filter/BM25/rerank."""
        query_emb = self._get_query_embedding(query.text)
        if query_emb is None:
            return [], trace

        results = self._dense.query(query_emb, top_k=self.config.k_final)
        if trace:
            trace.candidates_after_dense = len(results)
        scored = self._build_scored(results, "dense")
        return scored, trace

    def _keyword_only(
        self, query: RetrievalQuery, trace: RetrievalTrace | None
    ) -> tuple[list[ScoredEntry], RetrievalTrace | None]:
        """Pure BM25 retrieval with metadata filter."""
        candidates = apply_metadata_filter(self._entries, query)
        candidate_indices = [self._entry_index[id(c)] for c in candidates]
        if trace:
            trace.candidates_after_filter = len(candidates)

        if not candidate_indices:
            return [], trace

        results = self._lexical.query(
            query.text,
            candidate_indices=candidate_indices,
            top_k=self.config.k_final,
        )
        scored = self._build_scored(results, "bm25")
        return scored, trace

    def _build_scored(
        self, results: list[tuple[int, float]], stage_name: str
    ) -> list[ScoredEntry]:
        """Build ScoredEntry list from a single-stage result."""
        return [
            ScoredEntry(
                entry=self._entries[idx],
                score=score,
                stage_scores={stage_name: score, "bm25": None, "rerank": None},
            )
            for idx, score in results
        ]

    def _build_scored_with_bm25(
        self,
        fused: list[tuple[int, float]],
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[ScoredEntry]:
        """Build ScoredEntry list with dense + bm25 stage scores."""
        dense_map = dict(dense_results)
        bm25_map = dict(bm25_results)
        return [
            ScoredEntry(
                entry=self._entries[idx],
                score=score,
                stage_scores={
                    "dense": dense_map.get(idx),
                    "bm25": bm25_map.get(idx),
                    "rerank": None,
                },
            )
            for idx, score in fused
        ]

    def _build_scored_full(
        self,
        reranked: list[tuple[int, float]],
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[ScoredEntry]:
        """Build ScoredEntry list with all stage scores."""
        dense_map = dict(dense_results)
        bm25_map = dict(bm25_results)
        return [
            ScoredEntry(
                entry=self._entries[idx],
                score=score,
                stage_scores={
                    "dense": dense_map.get(idx),
                    "bm25": bm25_map.get(idx),
                    "rerank": score,
                },
            )
            for idx, score in reranked
        ]
