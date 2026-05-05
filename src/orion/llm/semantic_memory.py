"""Semantic memory K^B for Agent B — operator-authored infrastructure knowledge.

Read-only at runtime. Entries are curated offline and stored as a JSON file.
Retrieval uses tag filtering + keyword scoring (no embedding API dependency).
When an embedding backend is available, cosine similarity replaces keyword
scoring — the interface is the same.

v6 Section 4.5 schema:
  (topic, slice_type_tag, topology_tag, content, version, author)

Contents per v6:
  - Tier conventions
  - Inter-domain link characteristics
  - Known-good partitioning patterns
  - Anti-patterns
  - Domain-specific properties
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class KBEntry:
    """A single entry in the K^B semantic memory store."""

    topic: str
    slice_type_tag: str
    topology_tag: str
    content: str
    version: str = "1.0"
    author: str = "operator"
    _embedding: list[float] | None = field(default=None, repr=False)


class SemanticMemory:
    """K^B: Operator-authored infrastructure knowledge for Agent B.

    Entries are loaded from a JSON file and never modified at runtime.
    Retrieval filters by slice_type_tag and topology_tag, then ranks by
    keyword overlap (or embedding similarity when available).

    When a RetrievalPipeline is attached (via from_json_with_pipeline),
    retrieval delegates to the 4-stage hybrid pipeline.
    """

    def __init__(self, entries: list[KBEntry] | None = None) -> None:
        self.entries: list[KBEntry] = entries or []
        self._pipeline = None

    @classmethod
    def from_json_with_pipeline(
        cls,
        path: Path,
        config: Any,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> SemanticMemory:
        """Load entries and attach a RetrievalPipeline for hybrid retrieval.

        Args:
            path: Path to JSON file.
            config: RetrievalConfig instance.
            embed_fn: Embedding function (texts -> embeddings).
        """
        from orion.retrieval import MemoryEntry, RetrievalPipeline

        instance = cls.from_json(path)

        # Convert KBEntry to MemoryEntry for the pipeline
        memory_entries = []
        for i, e in enumerate(instance.entries):
            me = MemoryEntry(
                entry_id=f"kb_{i}",
                topic=e.topic,
                content=e.content,
                tags={
                    "slice_type": [e.slice_type_tag],
                    "topology": [e.topology_tag],
                },
            )
            memory_entries.append(me)

        pipeline = RetrievalPipeline(config, embed_fn)
        pipeline.build(memory_entries)
        instance._pipeline = pipeline
        return instance

    @classmethod
    def from_json(cls, path: Path) -> SemanticMemory:
        """Load entries from a JSON file.

        Expected format: list of objects with keys matching KBEntry fields.
        """
        with open(path) as f:
            raw = json.load(f)
        entries = [
            KBEntry(
                topic=e["topic"],
                slice_type_tag=e.get("slice_type_tag", "all"),
                topology_tag=e.get("topology_tag", "all"),
                content=e["content"],
                version=e.get("version", "1.0"),
                author=e.get("author", "operator"),
            )
            for e in raw
        ]
        return cls(entries=entries)

    def retrieve(
        self,
        query: str,
        slice_type: str | None = None,
        topology_tag: str | None = None,
        top_k: int = 5,
    ) -> list[KBEntry]:
        """Retrieve the top-k most relevant entries.

        When a pipeline is attached, delegates to the 4-stage hybrid retrieval.
        Otherwise uses tag filtering + keyword scoring.

        Args:
            query: The query text (typically a summary of the slice request).
            slice_type: Filter to entries matching this slice type (e.g. "eMBB").
            topology_tag: Filter to entries matching this topology tag.
            top_k: Maximum number of entries to return.

        Returns:
            List of KBEntry sorted by relevance (most relevant first).
        """
        if self._pipeline is not None:
            return self._retrieve_via_pipeline(query, slice_type, topology_tag, top_k)

        candidates = self._filter(slice_type, topology_tag)
        if not candidates:
            return []

        scored = [(e, self._score(query, e)) for e in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def _retrieve_via_pipeline(
        self,
        query: str,
        slice_type: str | None,
        topology_tag: str | None,
        top_k: int,
    ) -> list[KBEntry]:
        """Delegate retrieval to the attached pipeline."""
        from orion.retrieval import RetrievalQuery

        filters: dict[str, str | list[str]] = {}
        if slice_type:
            filters["slice_type"] = [slice_type, "all"]
        if topology_tag:
            filters["topology"] = [topology_tag, "all"]

        rq = RetrievalQuery(text=query, filters=filters, top_k=top_k)
        scored_entries, _ = self._pipeline.retrieve(rq)

        # Map back to KBEntry by index
        results = []
        for se in scored_entries:
            idx = int(se.entry.entry_id.replace("kb_", ""))
            if idx < len(self.entries):
                results.append(self.entries[idx])
        return results

    def format_for_prompt(self, entries: list[KBEntry]) -> str:
        """Format retrieved entries as a Reference Knowledge block for the prompt.

        Distinct from the Past Plans block populated by M^B.
        """
        if not entries:
            return ""

        parts = ["--- Reference Knowledge (Infrastructure) ---"]
        for i, e in enumerate(entries, 1):
            parts.append(f"\n[{i}] {e.topic}")
            parts.append(e.content)
        return "\n".join(parts)

    def _filter(
        self,
        slice_type: str | None,
        topology_tag: str | None,
    ) -> list[KBEntry]:
        """Filter entries by slice_type_tag and topology_tag."""
        result = []
        for e in self.entries:
            # slice_type filter
            if slice_type and e.slice_type_tag not in ("all", slice_type):
                continue
            # topology filter
            if topology_tag and e.topology_tag not in ("all", topology_tag):
                continue
            result.append(e)
        return result

    def _score(self, query: str, entry: KBEntry) -> float:
        """Score an entry against a query by keyword overlap.

        Returns a value in [0, 1] representing the fraction of query tokens
        found in the entry's topic + content.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0.0

        entry_tokens = _tokenize(entry.topic + " " + entry.content)
        overlap = query_tokens & entry_tokens
        return len(overlap) / len(query_tokens)


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def build_query_from_slice(slice_request: dict) -> str:
    """Build a retrieval query string from a slice request dict.

    Combines slice type, VNF types, tier requirements, and QoS values
    into a text query for keyword/embedding matching.
    """
    parts = [slice_request.get("slice_type", "")]

    for vnf in slice_request.get("vnfs", []):
        parts.append(vnf.get("vnf_type", ""))
        for tier in vnf.get("permitted_tiers", []):
            parts.append(tier)

    qos = slice_request.get("qos", {})
    if "max_e2e_delay" in qos:
        delay = qos["max_e2e_delay"]
        if delay <= 10:
            parts.append("ultra-low latency")
        elif delay <= 50:
            parts.append("low latency")
        else:
            parts.append("delay tolerant")

    if "min_throughput" in qos:
        tp = qos["min_throughput"]
        if tp >= 500:
            parts.append("high throughput")
        elif tp >= 100:
            parts.append("moderate throughput")

    return " ".join(parts)
