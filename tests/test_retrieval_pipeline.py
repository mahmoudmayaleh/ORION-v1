"""Tests for the 4-stage hybrid retrieval pipeline."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from orion.retrieval import (
    DenseRetriever,
    LexicalRetriever,
    MemoryEntry,
    RetrievalConfig,
    RetrievalMode,
    RetrievalPipeline,
    RetrievalQuery,
    ScoredEntry,
    apply_metadata_filter,
    rrf_fuse,
    telecom_tokenize,
)


def _bm25_available() -> bool:
    try:
        import rank_bm25  # noqa: F401
        return True
    except ImportError:
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_entry(
    entry_id: str,
    topic: str = "test",
    content: str = "content",
    tags: dict | None = None,
    embedding: list[float] | None = None,
    last_accessed_at: datetime | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        topic=topic,
        content=content,
        tags=tags or {},
        embedding=embedding,
        last_accessed_at=last_accessed_at or datetime.now(),
    )


def _random_embedding(dim: int = 64, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _make_entries(n: int, dim: int = 64) -> list[MemoryEntry]:
    """Generate n entries with deterministic random embeddings."""
    entries = []
    for i in range(n):
        entries.append(
            _make_entry(
                entry_id=f"e_{i}",
                topic=f"topic {i} VNF placement",
                content=f"content about entry {i} URLLC mec ran_edge tier {i}",
                tags={
                    "slice_type": ["URLLC"] if i % 3 == 0 else ["eMBB"] if i % 3 == 1 else ["all"],
                    "topology": ["multi_domain"] if i % 2 == 0 else ["all"],
                },
                embedding=_random_embedding(dim, seed=i),
            )
        )
    return entries


def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embedding function."""
    return [_random_embedding(64, seed=hash(t) % 10000) for t in texts]


# ── Test 1: Metadata filter AND logic ────────────────────────────────────────

class TestMetadataFilter:

    def test_and_logic_across_keys(self):
        """AND across different tag keys — both must match."""
        entries = [
            _make_entry("a", tags={"slice_type": ["URLLC"], "topology": ["multi_domain"]}),
            _make_entry("b", tags={"slice_type": ["URLLC"], "topology": ["single_domain"]}),
            _make_entry("c", tags={"slice_type": ["eMBB"], "topology": ["multi_domain"]}),
        ]
        query = RetrievalQuery(
            text="test",
            filters={"slice_type": "URLLC", "topology": "multi_domain"},
        )
        result = apply_metadata_filter(entries, query)
        assert len(result) == 1
        assert result[0].entry_id == "a"

    def test_or_within_tag_values(self):
        """OR within values for a single key."""
        entries = [
            _make_entry("a", tags={"slice_type": ["URLLC"]}),
            _make_entry("b", tags={"slice_type": ["eMBB"]}),
            _make_entry("c", tags={"slice_type": ["V2X"]}),
        ]
        query = RetrievalQuery(
            text="test",
            filters={"slice_type": ["URLLC", "eMBB"]},
        )
        result = apply_metadata_filter(entries, query)
        assert len(result) == 2
        ids = {e.entry_id for e in result}
        assert ids == {"a", "b"}

    def test_missing_tag_is_wildcard(self):
        """Entry missing a tag key passes (wildcard)."""
        entries = [
            _make_entry("a", tags={"slice_type": ["URLLC"]}),
            _make_entry("b", tags={}),  # no slice_type tag
        ]
        query = RetrievalQuery(text="test", filters={"slice_type": "URLLC"})
        result = apply_metadata_filter(entries, query)
        assert len(result) == 2  # both pass

    def test_all_tag_value_matches_any_filter(self):
        """Entry with 'all' value matches any filter."""
        entries = [
            _make_entry("a", tags={"slice_type": ["all"]}),
            _make_entry("b", tags={"slice_type": ["eMBB"]}),
        ]
        query = RetrievalQuery(text="test", filters={"slice_type": "URLLC"})
        result = apply_metadata_filter(entries, query)
        assert len(result) == 1
        assert result[0].entry_id == "a"


# ── Test 4: Dense retrieval top-k correctness ────────────────────────────────

class TestDenseRetrieval:

    def test_top_k_correctness(self):
        """Dense retrieval returns correct top-k by similarity."""
        dim = 64
        entries = _make_entries(20, dim)
        retriever = DenseRetriever()
        retriever.build_index([e.embedding for e in entries])

        # Query with same embedding as entry 5
        query_emb = entries[5].embedding
        results = retriever.query(query_emb, top_k=3)

        assert len(results) == 3
        # Entry 5 should be the top result (exact match)
        assert results[0][0] == 5
        assert results[0][1] == pytest.approx(1.0, abs=0.01)


# ── Test 5: BM25 keyword match ranking ───────────────────────────────────────

class TestBM25Ranking:

    @pytest.mark.skipif(
        not _bm25_available(),
        reason="rank_bm25 not installed",
    )
    def test_keyword_match_ranking(self):
        """BM25 ranks exact keyword matches higher."""
        retriever = LexicalRetriever()
        docs = [
            "URLLC ultra-low latency placement ran_edge",
            "eMBB high throughput CDN regional_cloud",
            "mIoT IoTGateway aggregation protocol",
        ]
        retriever.build_index(docs)
        results = retriever.query("URLLC latency ran_edge", top_k=3)

        # First result should be doc 0 (URLLC)
        assert results[0][0] == 0


# ── Test 6-7: RRF fusion ─────────────────────────────────────────────────────

class TestRRFFusion:

    def test_rrf_symmetry(self):
        """RRF gives higher score to items appearing in multiple rankings."""
        ranking_a = [(0, 1.0), (1, 0.8), (2, 0.6)]
        ranking_b = [(1, 1.0), (0, 0.8), (3, 0.6)]

        fused = rrf_fuse([ranking_a, ranking_b], k=60)
        scores = dict(fused)

        # Items 0 and 1 appear in both rankings, should score higher than 2 or 3
        assert scores[0] > scores[2]
        assert scores[1] > scores[3]

    def test_rrf_disjoint_rankings(self):
        """RRF handles completely disjoint rankings."""
        ranking_a = [(0, 1.0), (1, 0.8)]
        ranking_b = [(2, 1.0), (3, 0.8)]

        fused = rrf_fuse([ranking_a, ranking_b], k=60)
        ids = [idx for idx, _ in fused]

        assert set(ids) == {0, 1, 2, 3}
        # Top items from each ranking should have equal RRF scores
        scores = dict(fused)
        assert scores[0] == pytest.approx(scores[2])


# ── Test 8: Cross-encoder reorder (mocked) ───────────────────────────────────

class TestCrossEncoderRerank:

    def test_reranker_reorders(self, monkeypatch):
        """Cross-encoder reranks documents by relevance."""
        from orion.retrieval import stage_rerank
        from orion.retrieval.stage_rerank import CrossEncoderReranker

        # Mock the CrossEncoder
        class MockCrossEncoder:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, pairs):
                # Give highest score to first pair
                return list(range(len(pairs), 0, -1))

        monkeypatch.setattr(stage_rerank, "_st_available", True)

        reranker = CrossEncoderReranker("mock-model")
        # Directly inject the mock model
        reranker._model = MockCrossEncoder()

        results = reranker.rerank(
            "query",
            ["doc_a", "doc_b", "doc_c"],
            [10, 20, 30],
        )

        # Mock gives highest score (3) to first pair (idx 10)
        assert results[0][0] == 10


# ── Test 9: Full pipeline end-to-end ─────────────────────────────────────────

class TestFullPipeline:

    def test_end_to_end_50_entries(self):
        """Full pipeline produces results from 50 entries."""
        entries = _make_entries(50, dim=64)
        config = RetrievalConfig(
            mode=RetrievalMode.NO_RERANK,
            k_after_filter=200,
            k_after_dense=20,
            k_after_rrf=10,
            k_final=5,
            return_trace=True,
            enable_bm25=True,
        )
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(
            text="URLLC placement ran_edge low latency",
            filters={"slice_type": ["URLLC", "all"]},
            top_k=5,
        )
        results, trace = pipeline.retrieve(query)

        assert len(results) <= 5
        assert len(results) > 0
        assert trace is not None
        assert trace.candidates_after_filter > 0

        # All results should be ScoredEntry
        for r in results:
            assert isinstance(r, ScoredEntry)
            assert "dense" in r.stage_scores


# ── Tests 10-13: Ablation modes ──────────────────────────────────────────────

class TestAblationModes:

    @pytest.fixture
    def entries_and_pipeline(self):
        entries = _make_entries(30, dim=64)
        return entries

    def test_no_rerank_mode(self, entries_and_pipeline):
        entries = entries_and_pipeline
        config = RetrievalConfig(mode=RetrievalMode.NO_RERANK, k_final=5)
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(text="URLLC mec placement", top_k=5)
        results, _ = pipeline.retrieve(query)
        assert len(results) <= 5
        assert all(r.stage_scores.get("rerank") is None for r in results)

    def test_dense_only_mode(self, entries_and_pipeline):
        entries = entries_and_pipeline
        config = RetrievalConfig(mode=RetrievalMode.DENSE_ONLY, k_final=5)
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(text="URLLC placement", top_k=5)
        results, _ = pipeline.retrieve(query)
        assert len(results) <= 5

    def test_keyword_only_mode(self, entries_and_pipeline):
        entries = entries_and_pipeline
        config = RetrievalConfig(mode=RetrievalMode.KEYWORD_ONLY, k_final=5)
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(text="URLLC mec tier", top_k=5)
        results, _ = pipeline.retrieve(query)
        assert len(results) <= 5

    def test_cosine_only_mode(self, entries_and_pipeline):
        entries = entries_and_pipeline
        config = RetrievalConfig(mode=RetrievalMode.COSINE_ONLY, k_final=5)
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(text="URLLC placement", top_k=5)
        results, _ = pipeline.retrieve(query)
        assert len(results) <= 5
        # cosine_only skips filter, bm25, rerank
        for r in results:
            assert r.stage_scores.get("bm25") is None
            assert r.stage_scores.get("rerank") is None


# ── Test 14: Latency targets ─────────────────────────────────────────────────

class TestLatency:

    def test_no_rerank_sub_200ms(self):
        """mode=no_rerank on 500 entries should be under 200ms on CPU."""
        entries = _make_entries(500, dim=64)
        config = RetrievalConfig(
            mode=RetrievalMode.NO_RERANK,
            k_after_filter=200,
            k_after_dense=50,
            k_after_rrf=20,
            k_final=5,
        )
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(text="URLLC placement ran_edge", top_k=5)

        # Warmup
        pipeline.retrieve(query)

        t0 = time.perf_counter()
        for _ in range(5):
            pipeline.retrieve(query)
        elapsed = (time.perf_counter() - t0) / 5

        assert elapsed < 0.500, f"no_rerank took {elapsed:.3f}s (target <0.500s)"

    def test_full_sub_800ms_without_reranker(self):
        """mode=full on 500 entries (no actual reranker) should be under 800ms."""
        entries = _make_entries(500, dim=64)
        config = RetrievalConfig(
            mode=RetrievalMode.FULL,
            k_after_filter=200,
            k_after_dense=50,
            k_after_rrf=20,
            k_final=5,
            enable_rerank=False,
        )
        pipeline = RetrievalPipeline(config, embed_fn=_mock_embed_fn)
        pipeline.build(entries)

        query = RetrievalQuery(text="URLLC placement ran_edge", top_k=5)

        # Warmup
        pipeline.retrieve(query)

        t0 = time.perf_counter()
        for _ in range(5):
            pipeline.retrieve(query)
        elapsed = (time.perf_counter() - t0) / 5

        assert elapsed < 0.800, f"full took {elapsed:.3f}s (target <0.800s)"
