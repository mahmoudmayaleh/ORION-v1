"""Episodic memory M^B for Agent B — past placement experiences.

Records successful and failed placement plans with their outcomes.
Retrieval uses the same hybrid pipeline with recency weighting enabled.
Selective write: only records episodes with significant learning signal.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from orion.retrieval import (
    MemoryEntry,
    RetrievalConfig,
    RetrievalMode,
    RetrievalPipeline,
    RetrievalQuery,
    ScoredEntry,
)

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """M^B: Episodic memory storing past placement plans and outcomes.

    Uses the 4-stage retrieval pipeline with recency weighting enabled.
    Episodes are selectively recorded based on learning signal strength.
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        max_entries: int = 500,
    ) -> None:
        if config is None:
            config = RetrievalConfig(
                mode=RetrievalMode.NO_RERANK,
                apply_recency=True,
                k_final=3,
            )
        self.config = config
        self._embed_fn = embed_fn
        self._max_entries = max_entries
        self._entries: list[MemoryEntry] = []
        self._pipeline = RetrievalPipeline(config, embed_fn)

    def record(
        self,
        slice_spec: dict[str, Any],
        plan: dict[str, Any],
        m_committed: float,
        constraints_violated: list[str],
        reward: float,
    ) -> bool:
        """Record an episode if it carries significant learning signal.

        Selective write criteria:
          - High reward (success worth remembering)
          - Constraint violations (failure to learn from)
          - Unusual slice characteristics

        Returns:
            True if the episode was recorded.
        """
        # Selective write: only record if strong signal
        if not self._should_record(reward, constraints_violated):
            return False

        success = len(constraints_violated) == 0 and reward > 0
        label = "success" if success else "failure"

        content_parts = [
            f"Slice: {json.dumps(slice_spec, default=str)}",
            f"Plan: {json.dumps(plan, default=str)}",
            f"Reward: {reward:.4f}",
            f"Committed cost: {m_committed:.2f}",
        ]
        if constraints_violated:
            content_parts.append(f"Violations: {', '.join(constraints_violated)}")

        topic = f"{label}: {slice_spec.get('slice_type', 'unknown')} placement"
        content = "\n".join(content_parts)

        entry = MemoryEntry(
            entry_id=str(uuid.uuid4()),
            topic=topic,
            content=content,
            tags={
                "label": [label],
                "slice_type": [slice_spec.get("slice_type", "unknown")],
            },
            created_at=datetime.now(),
            last_accessed_at=datetime.now(),
        )

        self._entries.append(entry)
        self._pipeline.add_entry(entry)

        # Evict if over capacity
        if len(self._entries) > self._max_entries:
            self._evict()

        return True

    def retrieve(
        self,
        query: str,
        filters: dict[str, str | list[str]] | None = None,
        top_k: int = 3,
    ) -> list[ScoredEntry]:
        """Retrieve relevant past episodes with recency-weighted scoring."""
        rq = RetrievalQuery(
            text=query,
            filters=filters or {},
            top_k=top_k,
        )
        scored, _ = self._pipeline.retrieve(rq)

        # Update access metadata
        for se in scored:
            self._pipeline.update_access(se.entry.entry_id)

        return scored

    def format_for_prompt(self, entries: list[ScoredEntry]) -> str:
        """Format retrieved episodes as a Past Plans block for the prompt.

        Labels episodes as positive examples or counter-examples.
        """
        if not entries:
            return ""

        parts = ["--- Past Plans (Episodic Memory) ---"]
        for i, se in enumerate(entries, 1):
            entry = se.entry
            label_tags = entry.tags.get("label", ["unknown"])
            label = label_tags[0] if label_tags else "unknown"

            if label == "success":
                prefix = "[Positive Example]"
            else:
                prefix = "[Counter-Example]"

            parts.append(f"\n{prefix} [{i}] {entry.topic}")
            parts.append(entry.content)

        return "\n".join(parts)

    def to_few_shot(self, entries: list[ScoredEntry]) -> list[dict]:
        """Convert retrieved episodes to Agent B's few-shot format.

        Parses the stored content to extract slice_spec and plan dicts.
        Entries that fail to parse are skipped.
        """
        results: list[dict] = []
        for se in entries:
            try:
                slice_dict = None
                plan_dict = None
                for line in se.entry.content.splitlines():
                    if line.startswith("Slice: "):
                        slice_dict = json.loads(line[len("Slice: "):])
                    elif line.startswith("Plan: "):
                        plan_dict = json.loads(line[len("Plan: "):])
                if slice_dict is None or plan_dict is None:
                    logger.warning(
                        "to_few_shot_missing_fields",
                        extra={"entry_id": se.entry.entry_id},
                    )
                    continue
                results.append({
                    "slice_request": slice_dict,
                    "placement_plan": plan_dict,
                })
            except (json.JSONDecodeError, AttributeError) as exc:
                logger.warning(
                    "to_few_shot_parse_failed",
                    extra={"entry_id": se.entry.entry_id, "error": str(exc)},
                )
                continue
        return results

    def save(self, path: Path) -> None:
        """Persist episodic memory to JSON."""
        data = [e.model_dump(mode="json") for e in self._entries]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: Path) -> None:
        """Load episodic memory from JSON and rebuild indices."""
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        self._entries = [MemoryEntry(**d) for d in data]
        self._pipeline.build(self._entries)

    def _should_record(self, reward: float, violations: list[str]) -> bool:
        """Determine if an episode has enough learning signal to store."""
        if violations:
            return True
        if reward >= 0.8:
            return True
        if reward <= -0.5:
            return True
        return False

    def _evict(self) -> None:
        """Evict entries with lowest importance score.

        Importance = recency_weight * reward_proxy * retrieval_frequency.
        """
        now = datetime.now()
        tau = self.config.recency_tau

        scored: list[tuple[int, float]] = []
        for i, entry in enumerate(self._entries):
            delta_days = (now - entry.last_accessed_at).total_seconds() / 86400.0
            recency = math.exp(-tau * delta_days)
            frequency = math.log1p(entry.access_count)
            importance = recency * (1.0 + frequency)
            scored.append((i, importance))

        scored.sort(key=lambda x: x[1])

        # Remove the least important 10%
        n_remove = max(1, len(self._entries) // 10)
        remove_indices = {idx for idx, _ in scored[:n_remove]}

        removed_ids = [self._entries[i].entry_id for i in remove_indices]
        self._entries = [e for i, e in enumerate(self._entries) if i not in remove_indices]

        for eid in removed_ids:
            self._pipeline.remove_entry(eid)
