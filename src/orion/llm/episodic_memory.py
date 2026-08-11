"""Episodic memory M^B for Agent B — plan-layer adaptation memory.

Records plan-layer decisions and their survival outcomes against topology
signatures. M^B learns which tier assignments and plan shapes survive on
which topology families — the mechanism for cross-topology adaptation.

Schema (repointed 2026-07-02; state key repointed 2026-07-27 for §Y):
  Primary record: (condition_signature, topology_signature, slice_spec, plan_shape,
                   survival, violation_tag)
  plan_shape includes: strategy label, tier assignment, cut points, inter-domain links
  Violation tags: C5b and C7 are first-class
  Retrieval keyed on slice spec features + network CONDITION (congestion/load),
  falling back to the topology signature for pre-§Y stores.

§Y.6: the state term used to be topology similarity. Under a single fixed
substrate that term is constant (every entry scores f_state = 1.0), which both
kills its ability to discriminate AND pins abstain_rate at 0, since
combined = 0.5*f_task + 0.5*1.0 >= RETRIEVAL_FLOOR for every candidate. The
state term now scores network condition (see llm/condition_signature.py), which
is what actually varies once the topology is fixed.

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

from orion.llm.condition_signature import condition_similarity
from orion.retrieval import (
    MemoryEntry,
    RetrievalConfig,
    RetrievalMode,
    RetrievalPipeline,
    RetrievalQuery,
    ScoredEntry,
    telecom_tokenize,
)

logger = logging.getLogger(__name__)

# ── Retrieval policy ─────────────────────────────────────────────────────────
# REMEMBERER (arXiv 2306.07929) weights task and state similarity equally at
# lambda=0.5; adopted unchanged rather than tuned, so the split is a citation
# and not a fitted parameter.
RETRIEVAL_LAMBDA = 0.5
# Abstain threshold on the combined score. Below it, retrieve nothing and let
# the planner run zero-shot.
#
# CALIBRATED 2026-07-29 against the §Y.6 condition key by
# scripts/probe_retrieval_floor.py (160 entries over 4 congestion regimes, 480
# scored hits). Measured combined-score distributions:
#
#   same-condition hits   n=120   min 0.625  p50 0.750  max 0.750
#   other-condition hits  n=360   min 0.193  p50 0.449  max 0.574
#
# The classes are disjoint, and 0.60 sits in the gap: it keeps every
# same-condition hit and drops every other-condition hit (Youden J = 1.000). The
# previous 0.5 admitted 88 wrong-condition hits.
#
# Caveat, so this is not over-read: the probe drains the substrate uniformly to
# four well-separated regimes, while a real stream produces a continuum. Perfect
# separation is therefore optimistic and the operating point should be re-read
# from `retrieval_stats()` on the first real run. What the probe does establish
# is the SCALE, which is the part that was wrong before.
#
# The prior comment claimed "CALIBRATED, NOT ASSUMED -- see
# scripts/probe_retrieval.py"; that script was not in the tree, and the
# distributions it referred to were same-TOPOLOGY vs cross-TOPOLOGY, which is not
# what the state term scores now.
RETRIEVAL_FLOOR = 0.60

# The floor is SCALE-DEPENDENT, so there are two of them.
#
# `combined` is on a different scale depending on whether a state term exists:
#   condition key present -> lambda*f_task + (1-lambda)*f_state, measured 0.19..0.75
#   no state term         -> f_task alone, BM25 containment, typically 0.02..0.30
# 0.60 is calibrated on the FIRST scale only. Applying it to the second abstains
# on essentially everything, which is not a stricter policy, it is a different
# measurement being compared to the wrong threshold.
#
# The legacy floor is kept for the topology / text-only paths so pre-§Y stores
# behave exactly as before. Those paths are deprecated under §Y.6 but must not be
# silently changed while they still exist.
RETRIEVAL_FLOOR_LEGACY = 0.5


class _AutoFloor:
    """Sentinel: pick the floor matching the scoring mode actually used.

    Distinct from None, which means "no floor at all" and is a legitimate
    caller choice (the calibration probe passes an explicit 0.0).
    """

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "AUTO_FLOOR"


AUTO_FLOOR = _AutoFloor()

# ExpeL (AAAI 2024): write everything, retrieve only successes as exemplars.
# Failures are retained in the store for offline analysis, never shown to the
# planner as a plan to copy.
RETRIEVE_SUCCESSES_ONLY = True
POOL_FACTOR = 8
MIN_POOL = 24


def _containment(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Fraction of QUERY tokens present in the document, in [0,1].

    Containment, not Jaccard: entries are long JSON blobs (~500 tokens) and the
    query is ~20, so a union-normalized Jaccard is capped near 0.04 and the text
    term dies -- measured, and it silently reduced ranking to topology alone.
    Absolute rather than corpus-relative (unlike raw BM25), so a fixed abstain
    floor is meaningful. Matches SemanticMemory._score.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _parse_tagged_json(content: str, prefix: str) -> dict:
    """Pull a `<prefix>: {...}` line back out of an entry's content."""
    for line in content.splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                return {}
    return {}


def _parse_topology(content: str) -> dict:
    """Pull the stored topology signature back out of an entry's content."""
    return _parse_tagged_json(content, "Topology: ")


def _parse_condition(content: str) -> dict:
    """Pull the stored network-condition signature back out (§Y.6)."""
    return _parse_tagged_json(content, "Condition: ")


def _topology_similarity(a: dict, b: dict) -> float:
    """Substrate similarity in [0,1] over the numeric signature fields.

    Deliberately numeric rather than lexical: the family code ('C+_T-_B-') is
    NOT usable as a retrieval key, because telecom_tokenize strips '+'/'-' and
    collapses every family to ['c','t','b'] -- so C+_T+_B- and C-_T-_B+ are
    textually identical. Matching on scale and scarcity instead is what makes
    the state term discriminate at all.
    """
    if not a or not b:
        return 0.0

    def _cov(d):
        tc = d.get("tier_coverage")
        return (sum(tc) / len(tc)) if isinstance(tc, list) and tc else None

    def _nodes(d):
        npd = d.get("nodes_per_domain")
        return float(sum(npd)) if isinstance(npd, list) and npd else None

    # (accessor, scale) — scale sets what counts as "totally different" so each
    # feature contributes comparably rather than by raw magnitude.
    feats = [
        (_cov, 1.0),
        (_nodes, 45.0),                                   # spans 15..60 nodes
        (lambda d: d.get("inter_bw_mean"), 900.0),        # spans ~50..1000
        (lambda d: float(d["num_domains"]) if d.get("num_domains") else None, 8.0),
        (lambda d: float(d["num_inter_links"]) if d.get("num_inter_links") else None, 40.0),
    ]
    sims = []
    for fn, scale in feats:
        va, vb = fn(a), fn(b)
        if va is None or vb is None:
            continue
        sims.append(max(0.0, 1.0 - abs(va - vb) / scale))
    return sum(sims) / len(sims) if sims else 0.0


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
        assert write_policy in ("selective", "write_all", "reward")
        assert evict_policy in ("importance", "fifo")
        self._write_policy = write_policy
        self._evict_policy = evict_policy
        self._last_retrieval: dict | None = None
        self._last_scores: list[float] = []
        self._last_abstained: bool = False
        self._retrieval_log: list[dict] = []
        self._entries: list[MemoryEntry] = []
        self._pipeline = RetrievalPipeline(config, embed_fn)

    def retrieval_stats(self) -> dict:
        """Aggregate retrieval behaviour since construction (or reset)."""
        lg = self._retrieval_log
        if not lg:
            return {"calls": 0}
        ab = sum(1 for r in lg if r["abstained"])
        fired = [r["max"] for r in lg if not r["abstained"] and r["max"] is not None]
        return {
            "calls": len(lg),
            "abstain_rate": round(ab / len(lg), 3),
            "mean_returned": round(sum(r["n"] for r in lg) / len(lg), 2),
            "mean_score_when_fired": round(sum(fired) / len(fired), 4) if fired else None,
        }

    def reset_retrieval_log(self) -> None:
        """Clear ONLY the telemetry. Must not touch _entries/_pipeline -- callers
        invoke this on a loaded store right before eval."""
        self._retrieval_log = []

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
        committed_partition: list[int] | None = None,
        diverged: bool | None = None,
        condition_signature: dict[str, Any] | None = None,
    ) -> bool:
        """Record an episode if it carries significant learning signal.

        committed_partition / diverged close the second outcome loop: the partition
        the RL coordinator ACTUALLY committed (which may differ from Agent B's
        suggestion) and whether it diverged. Retrieval surfaces these so future
        plans are steered toward what the coordinator was observed to commit.

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
            condition_signature: Network condition at decision time (§Y.6) —
                per-domain / per-tier residual fractions, inter-domain headroom,
                load level. This is the retrieval STATE key under a fixed
                topology; see llm/condition_signature.py.

        Returns:
            True if the episode was recorded.
        """
        if self._write_policy == "selective" and not self._should_record(
            reward, constraints_violated
        ):
            return False
        if self._write_policy == "reward" and not self._should_record_reward(reward):
            return False

        if self._write_policy == "reward":
            label = "success" if reward > 0 else "failure"
        else:
            success = len(constraints_violated) == 0 and reward > 0
            label = "success" if success else "failure"

        content_parts = []

        # Network condition at decision time (§Y.6 retrieval state key).
        if condition_signature:
            content_parts.append(f"Condition: {json.dumps(condition_signature, default=str)}")

        # Topology signature (pre-§Y state key; retained for legacy stores).
        if topology_signature:
            content_parts.append(f"Topology: {json.dumps(topology_signature, default=str)}")

        content_parts.append(f"Slice: {json.dumps(slice_spec, default=str)}")

        # Plan shape (strategy, tier assignment, cut points, inter-domain links)
        if plan_shape:
            content_parts.append(f"PlanShape: {json.dumps(plan_shape, default=str)}")
        content_parts.append(f"Plan: {json.dumps(plan, default=str)}")

        content_parts.append(f"Reward: {reward:.4f}")
        content_parts.append(f"Committed cost: {m_committed:.2f}")

        # Second outcome loop: what the RL coordinator actually committed.
        if committed_partition is not None:
            content_parts.append(
                f"CommittedPartition: {json.dumps(committed_partition, default=str)}")
        if diverged is not None:
            content_parts.append(f"Diverged: {diverged}")

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
        if diverged is not None:
            tags["diverged"] = ["yes" if diverged else "no"]
        if condition_signature:
            # Filterable operating point: a caller can restrict retrieval to the
            # same congestion regime without touching the ranking terms.
            if condition_signature.get("bucket"):
                tags["congestion"] = [condition_signature["bucket"]]
            if condition_signature.get("load_level"):
                tags["load_level"] = [condition_signature["load_level"]]

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
        topology: dict | None = None,
        successes_only: bool = RETRIEVE_SUCCESSES_ONLY,
        min_score: float | None | _AutoFloor = AUTO_FLOOR,
        condition: dict | None = None,
    ) -> list[ScoredEntry]:
        """Retrieve relevant past episodes.

        Three properties, each following published practice, because retrieving a
        near-miss episode is measurably worse than retrieving nothing (Cuconasu
        et al., SIGIR 2024: near-but-wrong retrievals hurt while random documents
        help; Xiong et al., arXiv 2505.16067: agents copy retrieved records with
        input/output correlation r ~ 1, so a topology-mismatched episode is
        replayed verbatim):

        1. State-aware ranking. Scoring is REMEMBERER's two-term form
           (arXiv 2306.07929), S = lambda*f_task + (1-lambda)*f_state: slice
           similarity AND state similarity, rather than slice text alone.
           Text-only ranking is what surfaces a same-request/different-state
           episode -- the exact negative-transfer case.

           `condition` (§Y.6) is the state key: network congestion and operating
           point. `topology` is the pre-§Y key and is used only when no condition
           is supplied. Under a fixed substrate the topology term is constant at
           1.0 for every candidate, which does not rank AND makes the abstain
           floor unreachable (0.5*f_task + 0.5*1.0 >= 0.5 always) -- so passing
           `topology` on a §Y run is a silent no-op, not a fallback.
        2. Successes only as exemplars (ExpeL, AAAI 2024). Failures stay in the
           store but are never handed to the planner as things to imitate.
        3. Abstain. Below the floor we return FEWER episodes, possibly zero, and
           the caller falls back to a zero-shot prompt.

        Both score terms are absolute in [0,1] -- deliberately not min-max
        normalized over the candidate pool, since min-max forces a maximum of 1.0
        and would make the floor unreachable, i.e. abstention impossible.
        """
        # Widen the candidate pool: post-filters below discard entries, and the
        # rerank reorders them, so pulling exactly top_k here would starve both.
        pool_k = max(top_k * POOL_FACTOR, MIN_POOL)
        rq = RetrievalQuery(
            text=query,
            filters=filters or {},
            top_k=pool_k,
        )
        scored, _ = self._pipeline.retrieve(rq)

        if successes_only:
            scored = [se for se in scored
                      if (se.entry.tags.get("label") or ["?"])[0] == "success"]

        q_tokens = set(telecom_tokenize(query.lower()))
        reranked: list[tuple[float, ScoredEntry]] = []
        for se in scored:
            f_task = _containment(q_tokens, set(telecom_tokenize(se.entry.content.lower())))
            if condition is not None:
                f_state = condition_similarity(condition, _parse_condition(se.entry.content))
            elif topology is not None:
                f_state = _topology_similarity(topology, _parse_topology(se.entry.content))
            else:
                f_state = None
            combined = f_task if f_state is None else (
                RETRIEVAL_LAMBDA * f_task + (1.0 - RETRIEVAL_LAMBDA) * f_state)
            reranked.append((combined, se))

        reranked.sort(key=lambda x: x[0], reverse=True)
        if isinstance(min_score, _AutoFloor):
            # Match the floor to the scale `combined` was actually computed on.
            min_score = (RETRIEVAL_FLOOR if condition is not None
                         else RETRIEVAL_FLOOR_LEGACY)
        if min_score is not None:
            reranked = [(s, se) for s, se in reranked if s >= min_score]
        kept = reranked[:top_k]
        self._last_scores = [round(s, 4) for s, _ in kept]
        self._last_abstained = not kept
        # Abstain rate is a first-class reported metric, not an inference: with a
        # floor, "memory was consulted" and "memory contributed" are different
        # events, and only the log distinguishes them.
        self._retrieval_log.append({
            "n": len(kept),
            "max": round(reranked[0][0], 4) if reranked else None,
            "best_available": round(max((s for s, _ in reranked), default=0.0), 4),
            "abstained": not kept,
        })
        # Return the score the FLOOR was applied to, not the raw pipeline score.
        # `retrieve` gates on `combined` but used to hand back entries carrying the
        # lexical stage score, which is on a different scale entirely (BM25
        # containment ~0.03 against a floor of 0.5). Any caller reading `.score`
        # was reading a number that had never been compared to the threshold, and
        # comparing the two would say retrieval was abstaining when it was not.
        # The pipeline score is preserved under stage_scores["pipeline"].
        scored = [
            se._replace(score=s,
                        stage_scores={**se.stage_scores, "pipeline": se.score,
                                      "combined": s})
            for s, se in kept
        ]

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
                    if line.startswith("Condition: "):
                        parsed["condition"] = json.loads(line[len("Condition: "):])
                    elif line.startswith("Topology: "):
                        parsed["topology"] = json.loads(line[len("Topology: "):])
                    elif line.startswith("Slice: "):
                        parsed["slice_request"] = json.loads(line[len("Slice: "):])
                    elif line.startswith("PlanShape: "):
                        parsed["plan_shape"] = json.loads(line[len("PlanShape: "):])
                    elif line.startswith("Plan: "):
                        parsed["placement_plan"] = json.loads(line[len("Plan: "):])
                    elif line.startswith("ViolationTag: "):
                        parsed["violation_tag"] = line[len("ViolationTag: "):]
                    elif line.startswith("CommittedPartition: "):
                        parsed["committed_partition"] = json.loads(
                            line[len("CommittedPartition: "):])
                    elif line.startswith("Diverged: "):
                        parsed["diverged"] = line[len("Diverged: "):].strip() == "True"
                # Outcome label (positive example vs counter-example) from the topic.
                parsed["outcome"] = ("feasible" if se.entry.topic.startswith("success")
                                     else "INFEASIBLE")

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

    def _should_record_reward(self, reward: float) -> bool:
        """Reward-labelled write gate (approach A): keyed on scalar admission reward only.

        BINARY on the sign, not a magnitude band. The previous form
        (reward >= 0.8 or reward <= -0.5) had no basis -- those cutoffs were
        invented, and the dead band silently discarded every episode in
        (-0.5, 0.8), which is most of the reward range.

        Binary outcome-gating is the published form: AWM (arXiv 2409.07429)
        admits only trajectories its evaluator labels L_eval = 1; Memp retains
        only successfully-completed trajectories. Xiong et al. (arXiv 2505.16067)
        supply the empirical case -- ungated add-all cost RegAgent 12.05 points
        versus writing nothing, while outcome-gated addition beat the no-memory
        baseline (70.95% vs 67.53%).
        """
        return reward > 0.0

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
