"""Type definitions for the 4-stage hybrid retrieval pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    """Pipeline execution mode for ablation control."""

    FULL = "full"
    NO_RERANK = "no_rerank"
    DENSE_ONLY = "dense_only"
    KEYWORD_ONLY = "keyword_only"
    COSINE_ONLY = "cosine_only"


class MemoryEntry(BaseModel):
    """A single entry in the retrieval corpus (K^B or M^B)."""

    entry_id: str
    topic: str
    content: str
    tags: dict[str, list[str]] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0


class RetrievalQuery(BaseModel):
    """A query submitted to the retrieval pipeline."""

    text: str
    filters: dict[str, str | list[str]] = Field(default_factory=dict)
    top_k: int = 5


class ScoredEntry(NamedTuple):
    """A retrieval result with per-stage scoring breakdown."""

    entry: MemoryEntry
    score: float
    stage_scores: dict[str, float | None]


class RetrievalTrace(BaseModel):
    """Debug trace of candidate counts through pipeline stages."""

    candidates_after_filter: int = 0
    candidates_after_dense: int = 0
    candidates_after_rrf: int = 0
    candidates_after_rerank: int = 0


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval pipeline."""

    mode: RetrievalMode = RetrievalMode.FULL
    k_after_filter: int = 200
    k_after_dense: int = 50
    k_after_rrf: int = 20
    k_final: int = 5
    rrf_k: int = 60
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    enable_bm25: bool = True
    enable_rerank: bool = True
    return_trace: bool = False
    recency_tau: float = 1.0 / 30.0
    apply_recency: bool = False
