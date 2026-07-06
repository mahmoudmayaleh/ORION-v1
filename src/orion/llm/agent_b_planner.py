"""Agent B plan builder: wires the LLM into the episode runner's plan_builder seam.

The episode runner consumes `plan_builder(slice_req, substrate) -> PlanSummary | None`.
This module provides `make_agent_b_plan_builder` which returns exactly that callable,
backed by the real Agent B with plan caching, structural check, and retry.

Three layers:
  1. `slice_request_to_dict` — convert SliceRequest (with permitted_nodes) into the
     dict Agent B expects (with permitted_tiers derived from the substrate).
  2. `plan_summary_from_agent_b` — map Agent B's vnf_assignments dict onto PlanSummary
     aligned to the slice's VNF order.
  3. `make_agent_b_plan_builder` — factory wiring (1)+(2) around AgentB.generate_and_check
     with plan caching (signature-keyed, stale-flag invalidation).

Quality instrumentation:
  `PlanQualityLog` records per-plan metrics for the falsifier the training side
  doesn't have: is this partition actually better than the heuristic default?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from orion.llm.abstract_topology import build_abstract_topology
from orion.llm.plan_cache import (
    AbstractPlan,
    PlanCache,
    instantiate_plan,
    plan_signature,
    revalidate_plan,
)
from orion.mdo.types import PlanSummary
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier, SliceRequest

if TYPE_CHECKING:
    from orion.llm.agent_b import AgentB
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.llm.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)

PlanBuilder = Callable[[SliceRequest, SubstrateNetwork], "PlanSummary | None"]


# ── Conversion helpers ───────────────────────────────────────────────────────


def _permitted_tiers(vnf, substrate: SubstrateNetwork) -> list[str]:
    """Derive tier-value strings from a VNF's permitted nodes (C8)."""
    g = substrate.graph
    return sorted({
        g.nodes[n]["tier"] for n in vnf.permitted_nodes if n in g.nodes
    })


def slice_request_to_dict(
    slice_req: SliceRequest, substrate: SubstrateNetwork
) -> dict:
    """Serialise a SliceRequest into the dict Agent B + structural checker consume."""
    return {
        "request_id": slice_req.request_id,
        "slice_type": slice_req.slice_type.value,
        "vnfs": [
            {
                "vnf_id": v.vnf_id,
                "vnf_type": v.vnf_type,
                "cpu_demand": v.cpu_demand,
                "ram_demand": v.ram_demand,
                "permitted_tiers": _permitted_tiers(v, substrate),
                "vcr": v.vcr,
            }
            for v in slice_req.vnfs
        ],
        "flow_edges": [
            {
                "source_vnf": e.source_vnf,
                "target_vnf": e.target_vnf,
                "bandwidth_demand": e.bandwidth_demand,
            }
            for e in slice_req.flow_edges
        ],
        "qos": {
            "max_e2e_delay": slice_req.qos.max_e2e_delay,
            "min_throughput": slice_req.qos.min_throughput,
        },
    }


def _domain_id_from_str(domain: object) -> int:
    """Parse Agent B's domain field ("d0" / 0 / "0") into an int."""
    if isinstance(domain, int):
        return domain
    s = str(domain).strip()
    if s.startswith("d") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    raise ValueError(f"Unparseable domain id from Agent B: {domain!r}")


def _tier_from_str(tier: object) -> InfrastructureTier:
    """Parse Agent B's required_tier value into an InfrastructureTier."""
    try:
        return InfrastructureTier(str(tier))
    except ValueError as exc:
        raise ValueError(
            f"Unknown required_tier from Agent B: {tier!r}"
        ) from exc


def abstract_plan_from_agent_b(
    plan: dict, slice_req: SliceRequest
) -> AbstractPlan:
    """Map Agent B's plan dict onto an AbstractPlan (the cacheable core)."""
    assignments = {a["vnf_id"]: a for a in plan.get("vnf_assignments", [])}

    required_tiers = []
    suggested_domains = []

    for vnf in slice_req.vnfs:
        a = assignments.get(vnf.vnf_id)
        if a is None:
            raise ValueError(
                f"Agent B plan missing assignment for VNF {vnf.vnf_id!r}"
            )
        required_tiers.append(_tier_from_str(a.get("required_tier")))
        suggested_domains.append(_domain_id_from_str(a.get("domain")))

    return AbstractPlan(
        sfc_template=tuple(v.vnf_type for v in slice_req.vnfs),
        required_tiers=required_tiers,
        suggested_domains=suggested_domains,
    )


def plan_summary_from_agent_b(
    plan: dict, slice_req: SliceRequest
) -> PlanSummary:
    """Map Agent B's plan dict onto PlanSummary (convenience wrapper)."""
    abstract = abstract_plan_from_agent_b(plan, slice_req)
    return instantiate_plan(abstract, slice_req)


# ── Quality instrumentation ─────────────────────────────────────────────────


@dataclass
class PlanQualityLog:
    """Per-plan quality metrics for Agent B output instrumentation.

    The falsifier the training side doesn't have: is Agent B's suggested
    partition strategically better than the heuristic default?
    """

    request_id: str
    slice_type: str
    # Agent B's partition
    agent_b_domains: list[int]
    # Heuristic default's partition (for comparison)
    heuristic_domains: list[int]
    # Do they differ?
    partitions_differ: bool
    # Number of inter-domain crossings in each
    agent_b_crossings: int
    heuristic_crossings: int
    # Structural validity
    structurally_valid: bool
    # Cache hit (no LLM call needed)
    cache_hit: bool = False


@dataclass
class PlanQualityTracker:
    """Accumulates PlanQualityLogs across an episode for aggregate stats.

    Tracks both successful plans AND fallbacks (Agent B returned invalid/
    unparseable output -> None -> D2 reject). A high fallback rate means
    the prior comparison is run on a biased subset of arrivals.
    """

    logs: list[PlanQualityLog] = field(default_factory=list)
    # Fallback counter: Agent B invoked but output was invalid/unparseable
    fallback_count: int = 0
    # Total Agent B invocations (cache misses that triggered an LLM call)
    llm_call_count: int = 0

    def record(self, log: PlanQualityLog) -> None:
        self.logs.append(log)
        if not log.cache_hit:
            self.llm_call_count += 1

    def record_fallback(self, request_id: str, reason: str) -> None:
        """Record an Agent B invocation that fell back to None (D2)."""
        self.fallback_count += 1
        self.llm_call_count += 1
        logger.info("Agent B fallback: %s — %s", request_id, reason)

    def summary(self) -> dict:
        if not self.logs and self.fallback_count == 0:
            return {}
        n_success = len(self.logs)
        n_total = n_success + self.fallback_count
        n_valid = sum(1 for l in self.logs if l.structurally_valid)
        n_differ = sum(1 for l in self.logs if l.partitions_differ)
        n_cache_hit = sum(1 for l in self.logs if l.cache_hit)
        ab_crossings = sum(l.agent_b_crossings for l in self.logs)
        h_crossings = sum(l.heuristic_crossings for l in self.logs)
        return {
            "total_plans": n_total,
            "successful_plans": n_success,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / max(n_total, 1),
            "llm_call_count": self.llm_call_count,
            "structurally_valid_rate": n_valid / max(n_success, 1),
            "partitions_differ_rate": n_differ / max(n_success, 1),
            "cache_hit_rate": n_cache_hit / max(n_success, 1),
            "agent_b_avg_crossings": ab_crossings / max(n_success, 1),
            "heuristic_avg_crossings": h_crossings / max(n_success, 1),
            "agent_b_fewer_crossings": sum(
                1 for l in self.logs
                if l.agent_b_crossings < l.heuristic_crossings
            ) / max(n_differ, 1),
        }


def _count_crossings(domains: list[int]) -> int:
    return sum(1 for i in range(len(domains) - 1) if domains[i] != domains[i + 1])


# ── Plan builder factory ────────────────────────────────────────────────────


def _heuristic_domains(slice_req: SliceRequest, substrate: SubstrateNetwork) -> list[int]:
    """Compute the heuristic default's suggested_domains for quality comparison."""
    from orion.sim.episode_runner import _default_plan_builder
    plan = _default_plan_builder(slice_req, substrate)
    if plan is None:
        return []
    return plan.suggested_domains


def make_agent_b_plan_builder(
    agent_b: "AgentB",
    *,
    kb: "SemanticMemory | None" = None,
    mb: "EpisodicMemory | None" = None,
    max_retries: int = 1,
    quality_tracker: PlanQualityTracker | None = None,
) -> PlanBuilder:
    """Build the Agent-B-backed plan_builder for EpisodeRunner.

    Returns a callable matching the plan_builder interface:
        (SliceRequest, SubstrateNetwork) -> PlanSummary | None

    Plan caching is signature-keyed: one LLM call per distinct
    (slice_type, qos_bucket), reused across arrivals of the same class.
    Persistent rejections trip the Page-Hinkley monitor -> stale -> refresh.

    Args:
        agent_b: The AgentB instance (owns the LLM backend).
        kb: Semantic memory K^B for reference knowledge.
        mb: Episodic memory M^B for few-shot examples.
        max_retries: Structural-check regeneration budget.
        quality_tracker: If provided, logs per-plan quality metrics.
    """
    # Bounded LRU cache keyed by plan_signature (slice_type, qos_bucket, sfc_template).
    # The finer key prevents the misaligned-partition bug: two requests with the
    # same (slice_type, qos_bucket) but different chain shapes get distinct entries.
    cache: PlanCache[AbstractPlan] = PlanCache(capacity=64)

    def build(
        slice_req: SliceRequest, substrate: SubstrateNetwork
    ) -> PlanSummary | None:
        sig = plan_signature(slice_req)

        # Cache lookup — returns the AbstractPlan (request-invariant core)
        entry = cache.get(sig)
        if entry is not None:
            abstract = entry.plan
            plan_summary = instantiate_plan(abstract, slice_req)

            # Revalidate: does the cached partition still fit the topology?
            if not revalidate_plan(plan_summary, substrate):
                cache.mark_stale(sig)
                # Fall through to Agent B invocation below
            else:
                if quality_tracker is not None:
                    h_domains = _heuristic_domains(slice_req, substrate)
                    quality_tracker.record(PlanQualityLog(
                        request_id=slice_req.request_id,
                        slice_type=slice_req.slice_type.value,
                        agent_b_domains=plan_summary.suggested_domains,
                        heuristic_domains=h_domains,
                        partitions_differ=plan_summary.suggested_domains != h_domains,
                        agent_b_crossings=_count_crossings(plan_summary.suggested_domains),
                        heuristic_crossings=_count_crossings(h_domains),
                        structurally_valid=True,
                        cache_hit=True,
                    ))
                return plan_summary

        # Cache miss or stale — invoke Agent B
        sr_dict = slice_request_to_dict(slice_req, substrate)
        topology = build_abstract_topology(substrate)

        try:
            if kb is not None or mb is not None:
                plan_dict, result = agent_b.generate_with_memory(
                    sr_dict, topology, kb=kb, mb=mb, max_retries=max_retries,
                )
            else:
                plan_dict, result = agent_b.generate_and_check(
                    sr_dict, topology, max_retries=max_retries,
                )
        except Exception:
            logger.exception(
                "Agent B call failed for %s — falling back to None (D2)",
                slice_req.request_id,
            )
            if quality_tracker is not None:
                quality_tracker.record_fallback(slice_req.request_id, "llm_call_exception")
            return None

        if not result.is_valid:
            logger.info(
                "Agent B plan infeasible for %s (%d violations) — D2 reject",
                slice_req.request_id, len(result.violations),
            )
            if quality_tracker is not None:
                quality_tracker.record_fallback(slice_req.request_id, "structural_check_failed")
            return None

        try:
            abstract = abstract_plan_from_agent_b(plan_dict, slice_req)
            plan_summary = instantiate_plan(abstract, slice_req)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Agent B plan parse failed for %s: %s — D2 reject",
                slice_req.request_id, exc,
            )
            if quality_tracker is not None:
                quality_tracker.record_fallback(slice_req.request_id, f"parse_error: {exc}")
            return None

        # Cache the AbstractPlan (not PlanSummary) under plan_signature
        cache.put(sig, abstract)

        if quality_tracker is not None:
            h_domains = _heuristic_domains(slice_req, substrate)
            quality_tracker.record(PlanQualityLog(
                request_id=slice_req.request_id,
                slice_type=slice_req.slice_type.value,
                agent_b_domains=plan_summary.suggested_domains,
                heuristic_domains=h_domains,
                partitions_differ=plan_summary.suggested_domains != h_domains,
                agent_b_crossings=_count_crossings(plan_summary.suggested_domains),
                heuristic_crossings=_count_crossings(h_domains),
                structurally_valid=True,
                cache_hit=False,
            ))

        return plan_summary

    return build


def make_constrained_plan_builder(
    model_path: str,
    *,
    n_threads: int = 32,
    n_ctx: int = 4096,
    quality_tracker: PlanQualityTracker | None = None,
) -> PlanBuilder:
    """Build a grammar-constrained Agent B plan builder.

    Uses ConstrainedAgentB with GBNF grammar to force valid JSON output.
    No K^B, no retries — the grammar prevents format failure.

    Args:
        model_path: Path to the GGUF model file.
        n_threads: CPU threads for llama-cpp inference.
        n_ctx: Context window size.
        quality_tracker: If provided, logs per-plan quality metrics.
    """
    from orion.llm.constrained_agent_b import ConstrainedAgentB

    constrained = ConstrainedAgentB(
        model_path=model_path, n_threads=n_threads, n_ctx=n_ctx,
    )
    cache: PlanCache[AbstractPlan] = PlanCache(capacity=64)

    def build(
        slice_req: SliceRequest, substrate: SubstrateNetwork,
    ) -> PlanSummary | None:
        sig = plan_signature(slice_req)

        # Cache lookup
        entry = cache.get(sig)
        if entry is not None:
            abstract = entry.plan
            plan_summary = instantiate_plan(abstract, slice_req)
            if revalidate_plan(plan_summary, substrate):
                if quality_tracker is not None:
                    h_domains = _heuristic_domains(slice_req, substrate)
                    quality_tracker.record(PlanQualityLog(
                        request_id=slice_req.request_id,
                        slice_type=slice_req.slice_type.value,
                        agent_b_domains=plan_summary.suggested_domains,
                        heuristic_domains=h_domains,
                        partitions_differ=plan_summary.suggested_domains != h_domains,
                        agent_b_crossings=_count_crossings(plan_summary.suggested_domains),
                        heuristic_crossings=_count_crossings(h_domains),
                        structurally_valid=True,
                        cache_hit=True,
                    ))
                return plan_summary
            cache.mark_stale(sig)

        # Cache miss — constrained Agent B call
        sr_dict = slice_request_to_dict(slice_req, substrate)
        topology = build_abstract_topology(substrate)

        try:
            plan_dict = constrained.generate_plan(sr_dict, topology)
        except Exception:
            logger.exception("Constrained Agent B failed for %s", slice_req.request_id)
            if quality_tracker is not None:
                quality_tracker.record_fallback(slice_req.request_id, "constrained_exception")
            return None

        if plan_dict is None:
            if quality_tracker is not None:
                quality_tracker.record_fallback(slice_req.request_id, "constrained_returned_none")
            return None

        try:
            abstract = abstract_plan_from_agent_b(plan_dict, slice_req)
            plan_summary = instantiate_plan(abstract, slice_req)
        except (ValueError, KeyError) as exc:
            logger.warning("Constrained plan parse failed: %s", exc)
            if quality_tracker is not None:
                quality_tracker.record_fallback(slice_req.request_id, f"parse: {exc}")
            return None

        cache.put(sig, abstract)

        if quality_tracker is not None:
            h_domains = _heuristic_domains(slice_req, substrate)
            quality_tracker.record(PlanQualityLog(
                request_id=slice_req.request_id,
                slice_type=slice_req.slice_type.value,
                agent_b_domains=plan_summary.suggested_domains,
                heuristic_domains=h_domains,
                partitions_differ=plan_summary.suggested_domains != h_domains,
                agent_b_crossings=_count_crossings(plan_summary.suggested_domains),
                heuristic_crossings=_count_crossings(h_domains),
                structurally_valid=True,
                cache_hit=False,
            ))

        return plan_summary

    return build
