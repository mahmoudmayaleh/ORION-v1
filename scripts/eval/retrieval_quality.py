"""Retrieval quality evaluation — Recall@k across all pipeline modes.

Usage:
    python scripts/eval/retrieval_quality.py

Loads tests/fixtures/retrieval_groundtruth.json and reports Recall@1, @3, @5
for each retrieval mode (full, no_rerank, dense_only, keyword_only, cosine_only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orion.retrieval import (
    MemoryEntry,
    RetrievalConfig,
    RetrievalMode,
    RetrievalPipeline,
    RetrievalQuery,
)


def compute_recall_at_k(
    ground_truth: list[dict],
    pipeline: RetrievalPipeline,
    k: int,
) -> float:
    """Compute Recall@k: fraction of queries where expected entry is in top-k."""
    hits = 0
    for gt in ground_truth:
        query = RetrievalQuery(
            text=gt["query_text"],
            filters={"slice_type": [gt["slice_type"], "all"]} if gt["slice_type"] != "all" else {},
            top_k=k,
        )
        results, _ = pipeline.retrieve(query)
        retrieved_ids = {r.entry.entry_id for r in results}
        if gt["expected_entry_id"] in retrieved_ids:
            hits += 1
    return hits / len(ground_truth) if ground_truth else 0.0


def load_kb_entries() -> list[MemoryEntry]:
    """Load kb_agent_b.json as MemoryEntry objects."""
    kb_path = PROJECT_ROOT / "data" / "memory" / "kb_agent_b.json"
    with open(kb_path) as f:
        raw = json.load(f)

    entries = []
    for i, e in enumerate(raw):
        entries.append(
            MemoryEntry(
                entry_id=f"kb_{i}",
                topic=e["topic"],
                content=e["content"],
                tags={
                    "slice_type": [e.get("slice_type_tag", "all")],
                    "topology": [e.get("topology_tag", "all")],
                },
            )
        )
    return entries


def mock_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embeddings for evaluation without GPU."""
    embeddings = []
    for t in texts:
        rng = np.random.default_rng(hash(t) % 2**31)
        vec = rng.standard_normal(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec.tolist())
    return embeddings


def main() -> None:
    gt_path = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_groundtruth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    entries = load_kb_entries()
    modes = [
        RetrievalMode.FULL,
        RetrievalMode.NO_RERANK,
        RetrievalMode.DENSE_ONLY,
        RetrievalMode.KEYWORD_ONLY,
        RetrievalMode.COSINE_ONLY,
    ]

    print(f"{'Mode':<15} {'Recall@1':>10} {'Recall@3':>10} {'Recall@5':>10}")
    print("-" * 50)

    for mode in modes:
        config = RetrievalConfig(
            mode=mode,
            k_after_filter=200,
            k_after_dense=20,
            k_after_rrf=10,
            k_final=5,
            enable_rerank=False,  # No reranker model in eval
        )
        pipeline = RetrievalPipeline(config, embed_fn=mock_embed_fn)
        pipeline.build(entries)

        r1 = compute_recall_at_k(ground_truth, pipeline, k=1)
        r3 = compute_recall_at_k(ground_truth, pipeline, k=3)
        r5 = compute_recall_at_k(ground_truth, pipeline, k=5)

        print(f"{mode.value:<15} {r1:>10.3f} {r3:>10.3f} {r5:>10.3f}")


if __name__ == "__main__":
    main()
