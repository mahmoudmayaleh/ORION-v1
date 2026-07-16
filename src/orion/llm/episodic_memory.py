"""Episodic memory M^B for Agent B — plan-layer adaptation memory.

Records plan-layer decisions and their survival outcomes against topology
signatures. M^B learns which tier assignments and plan shapes survive on
which topology families — the mechanism for cross-topology adaptation.

Schema (repointed 2026-07-02):
  Primary record: (topology_signature, slice_spec, plan_shape, survival, violation_tag)
  plan_shape includes: strategy label, tier assignment, cut points, inter-domain links
  Violation tags: C5b and C7 are first-class
  Retrieval keyed on topology signature + slice spec features

Routing-critical recoveries store split structure and inter-domain links used,
since that is precisely what distinguishes them from search failures and what
retrieval needs to reproduce them on similar topology signatures.
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
        write_policy: str = "selective",
        evict_policy: str = "importance",
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
        # Full-M^B: selective write + importance eviction (defaults).
        # FIFO-M^B ablation: write_policy="write_all", evict_policy="fifo".
        assert write_policy in ("selective", "write_all")
        assert evict_policy in ("importance", "fifo")
        self._write_policy = write_policy
        self._evict_policy = evict_policy
        self._last_retrieval: dict | None = None
        self._entries: list[MemoryEntry] = []
        self._pipeline = RetrievalPipeline(config, embed_fn)

    def record(
        self,
        slice_spec: dict[str, Any],
        plan: dict[str, Any],
        m_committed: float,
        constraints_violated: list[str],
        reward: float,
        topology_signature: dict[str, Any] | None = None,
        plan_shape: dict[str, Any] | None = None,
        violation_tag: str | None = None,
    ) -> bool:
        """Record an episode if it carries significant learning signal.

        Args:
            slice_spec: Slice request features (type, VNF count, tier requirements).
            plan: The placement plan produced by Agent B.
            m_committed: Committed cost of the plan.
            constraints_violated: List of constraint codes that fired.
            reward: Terminal reward for this episode.
            topology_signature: Topology features for retrieval keying
                (tier coverage, CPU/domain, inter-BW stats, connectivity).
            plan_shape: Plan strategy details for routing-critical recoveries:
                - strategy: "co-locate" | "split"
                - tier_assignment: per-VNF tier labels
                - cut_points: which VNF pairs cross domains
                - inter_domain_links: link IDs used for cross-domain flows
            violation_tag: First-class violation label (e.g., "C5b", "C7",
                "actor_infeasible"). None for successful admissions.

        Returns:
            True if the episode was recorded.
        """
        if self._write_policy == "selective" and not self._should_record(
            reward, constraints_violated
        ):
            return False

        success = len(constraints_violated) == 0 and reward > 0
        label = "success" if success else "failure"

        content_parts = []

        # Topology signature (primary retrieval key)
        if topology_signature:
            content_parts.append(f"Topology: {json.dumps(topology_signature, default=str)}")

        content_parts.append(f"Slice: {json.dumps(slice_spec, default=str)}")

        # Plan shape (strategy, tier assignment, cut points, inter-domain links)
        if plan_shape:
            content_parts.append(f"PlanShape: {json.dumps(plan_shape, default=str)}")
        content_parts.append(f"Plan: {json.dumps(plan, default=str)}")

        content_parts.append(f"Reward: {reward:.4f}")
        content_parts.append(f"Committed cost: {m_committed:.2f}")

        if constraints_violated:
            content_parts.append(f"Violations: {', '.join(constraints_violated)}")
        if violation_tag:
            content_parts.append(f"ViolationTag: {violation_tag}")

        topic = f"{label}: {slice_spec.get('slice_type', 'unknown')} placement"
        content = "\n".join(content_parts)

        # Tags for filtering — violation type is first-class
        tags: dict[str, list[str]] = {
            "label": [label],
            "slice_type": [slice_spec.get("slice_type", "unknown")],
        }
        if violation_tag:
            tags["violation"] = [violation_tag]
        if plan_shape and plan_shape.get("strategy"):
            tags["strategy"] = [plan_shape["strategy"]]

        entry = MemoryEntry(
            entry_id=str(uuid.uuid4()),
            topic=topic,
            content=content,
            tags=tags,
            created_at=datetime.now(),
            last_accessed_at=datetime.now(),
        )

        self._entries.append(entry)
        self._pipeline.add_entry(entry)

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

        # §P read-only telemetry: composition of the last retrieval (count + the
        # success/failure label mix of what was returned).
        n_pos = sum(1 for se in scored
                    if (se.entry.tags.get("label") or ["?"])[0] == "success")
        self._last_retrieval = {"n": len(scored), "pos": n_pos,
                                "neg": len(scored) - n_pos}

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

        Parses the stored content to extract slice_spec, plan, plan_shape,
        topology signature, and violation info. Entries that fail to parse
        are skipped.
        """
        results: list[dict] = []
        for se in entries:
            try:
                parsed: dict[str, Any] = {}
                for line in se.entry.content.splitlines():
                    if line.startswith("Topology: "):
                        parsed["topology"] = json.loads(line[len("Topology: "):])
                    elif line.startswith("Slice: "):
                        parsed["slice_request"] = json.loads(line[len("Slice: "):])
                    elif line.startswith("PlanShape: "):
                        parsed["plan_shape"] = json.loads(line[len("PlanShape: "):])
                    elif line.startswith("Plan: "):
                        parsed["placement_plan"] = json.loads(line[len("Plan: "):])
                    elif line.startswith("ViolationTag: "):
                        parsed["violation_tag"] = line[len("ViolationTag: "):]

                if "slice_request" not in parsed or "placement_plan" not in parsed:
                    logger.warning(
                        "to_few_shot_missing_fields",
                        extra={"entry_id": se.entry.entry_id},
                    )
                    continue
                results.append(parsed)
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

    def _create_entry(
        self,
        slice_spec: dict[str, Any],
        plan: dict[str, Any],
        m_committed: float,
        constraints_violated: list[str],
        reward: float,
        topology_signature: dict[str, Any] | None = None,
        plan_shape: dict[str, Any] | None = None,
        violation_tag: str | None = None,
    ) -> MemoryEntry:
        """Create a MemoryEntry without filtering. Used by FIFO write-all."""
        success = len(constraints_violated) == 0 and reward > 0
        label = "success" if success else "failure"

        content_parts = []
        if topology_signature:
            content_parts.append(f"Topology: {json.dumps(topology_signature, default=str)}")
        content_parts.append(f"Slice: {json.dumps(slice_spec, default=str)}")
        if plan_shape:
            content_parts.append(f"PlanShape: {json.dumps(plan_shape, default=str)}")
        content_parts.append(f"Plan: {json.dumps(plan, default=str)}")
        content_parts.append(f"Reward: {reward:.4f}")
        content_parts.append(f"Committed cost: {m_committed:.2f}")
        if constraints_violated:
            content_parts.append(f"Violations: {', '.join(constraints_violated)}")
        if violation_tag:
            content_parts.append(f"ViolationTag: {violation_tag}")

        tags: dict[str, list[str]] = {
            "label": [label],
            "slice_type": [slice_spec.get("slice_type", "unknown")],
        }
        if violation_tag:
            tags["violation"] = [violation_tag]
        if plan_shape and plan_shape.get("strategy"):
            tags["strategy"] = [plan_shape["strategy"]]

        return MemoryEntry(
            entry_id=str(uuid.uuid4()),
            topic=f"{label}: {slice_spec.get('slice_type', 'unknown')} placement",
            content="\n".join(content_parts),
            tags=tags,
            created_at=datetime.now(),
            last_accessed_at=datetime.now(),
        )

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
        """Evict entries. Importance policy (Full-M^B) removes lowest
        recency*reward*frequency; FIFO policy (FIFO-M^B ablation) removes
        oldest-first (insertion order = front of the list)."""
        n_remove = max(1, len(self._entries) // 10)

        if self._evict_policy == "fifo":
            remove_indices = set(range(n_remove))
        else:
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
            remove_indices = {idx for idx, _ in scored[:n_remove]}

        removed_ids = [self._entries[i].entry_id for i in remove_indices]
        self._entries = [e for i, e in enumerate(self._entries) if i not in remove_indices]

        for eid in removed_ids:
            self._pipeline.remove_entry(eid)
