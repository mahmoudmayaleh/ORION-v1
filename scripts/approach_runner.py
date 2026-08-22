#!/usr/bin/env python3
"""Three-approach learning-curve runner — implements the pre-registered protocol.

See docs/EXPERIMENT_PROTOCOL.md for the frozen specification.

Approaches:
  1. RA-ColocFB    — routability-aware co-location (static, no LLM)
  2. Memory-off    — Agent B + K^B, no M^B
  3. Full-M^B      — Agent B + K^B + M^B (selective write, importance eviction, K=50)

(FIFO-M^B and Plain-ColocFB were dropped from the original five-approach design.)

Usage:
  python scripts/approach_runner.py                    # real LLM (requires backend)
  python scripts/approach_runner.py --mock-llm         # mock Agent B (FFD-based, for infra testing)
  python scripts/approach_runner.py --mock-llm --seed 0  # single seed
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.baselines.colocation_ffd import (
    colocation_ffd,
    routability_aware_colocation_ffd,
    _select_node_in_domain,
)
from orion.baselines.greedy_ffd import (
    GreedyConfig, GreedyResult, _PlacementState, _run_greedy_ffd,
)
from orion.types import PlacementPlan
from orion.sim.qos_gate import plan_qos_ok
from orion.llm.abstract_topology import build_abstract_topology
from orion.llm.episodic_memory import EpisodicMemory
from orion.retrieval import RetrievalConfig, RetrievalMode
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.types import InfrastructureTier, SliceRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Protocol constants (frozen, from EXPERIMENT_PROTOCOL.md) ────────────────

MEMORY_CAPACITY_K = 50
TRAIN_INSTANCE_SEEDS = [0, 1]
TEST_INSTANCE_SEEDS = [0, 1]
INTERP_INSTANCE_SEED = 2
ARRIVALS_PER_INSTANCE = 100
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
RUN_SEEDS = [42, 43, 44]
N_STRUCT = 1  # No retries, 1 LLM call per arrival

# §P (2026-07-14): the pre-§Y held-out split. The topology families it refers to
# were removed in §Y.1e; kept only to date the warm-up convention below. Warm-up
# was 4 train topologies with 4 held out. Under §Y.1e it is training INSTANCES of
# one generator, held out by instance seed.
WARM_UP_ORDER = [
    "C+_T+_B+", "C-_T-_B+", "C-_T-_B-", "C+_T-_B+",
]

# WP8 planning-layer ablation (greedy executor, NO RL): Plain-ColocFB and the
# two LLM-partition approaches all share the SAME plain co-location fill (held fixed),
# so the figure isolates partition quality. RA-ColocFB is a REFERENCE line only
# (full-information oracle: reads global CPU residuals + all inter-domain BW +
# cross-domain path routing — not a deployable peer).
APPROACH_NAMES = ["RA-ColocFB", "Plain-ColocFB", "Memory-off", "FIFO-M^B", "Full-M^B"]
# Each LLM approach runs against its own llama.cpp server (own port) so the approaches
# execute concurrently. Keep in sync with the servers the supervisor launches.
APPROACH_LLM_PORTS = {"Memory-off": 8000, "FIFO-M^B": 8002, "Full-M^B": 8001}
STATIC_APPROACHES = {"RA-ColocFB", "Plain-ColocFB"}
# Approaches that write to an M^B store. FIFO-M^B = write-all + FIFO eviction
# (§4.1 ablation); Full-M^B = selective write + importance eviction.
MB_APPROACHES = {"Full-M^B", "FIFO-M^B"}
LLM_APPROACHES = {"Memory-off", "FIFO-M^B", "Full-M^B"}


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class InstanceResult:
    """Per-instance, per-approach result."""
    family: str
    instance_seed: int
    approach: str
    phase: str  # "warmup", "extrap", "interp"
    admitted: int = 0
    total: int = 0
    ceiling: int = 0
    fraction_of_ceiling: float = 0.0
    reject_reasons: dict = field(default_factory=dict)


# ── Ceiling computation (enumerator, executor-independent) ──────────────────



#: Node count above which the ceiling is refused outright. §Y substrates start at
#: 100 nodes; the enumerator is a pre-§Y instrument and its answer is
#: contention-blind, so running it there would be slow AND wrong rather than just


def plan_to_summary(plan_result, sr, substrate):
    """Convert a GreedyResult to PlanSummary for the coordinator."""
    from orion.mdo.types import PlanSummary
    if not plan_result.feasible or plan_result.plan is None:
        return None
    vnf_ids, required_tiers, suggested_domains = [], [], []
    g = substrate.graph
    for vnf in sr.vnfs:
        nid = plan_result.plan.vnf_placements[vnf.vnf_id]
        dom = g.nodes[nid]["domain_id"]
        tier = g.nodes[nid]["tier"]
        vnf_ids.append(vnf.vnf_id)
        required_tiers.append(InfrastructureTier(tier))
        suggested_domains.append(dom)
    return PlanSummary(
        vnf_ids=vnf_ids, required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in sr.vnfs],
        ram_demands=[v.ram_demand for v in sr.vnfs],
        vcrs=[v.vcr for v in sr.vnfs],
        bw_demands=[f.bandwidth_demand for f in sr.flow_edges],
    )


def run_static_approach(approach_name, substrate, sr):
    """Run a static approach: returns (admitted: bool, plan_summary_or_none)."""
    cfg = GreedyConfig()
    if approach_name == "RA-ColocFB":
        result = routability_aware_colocation_ffd(substrate, sr, cfg)
    elif approach_name == "Plain-ColocFB":
        result = colocation_ffd(substrate, sr, cfg)
    else:
        raise ValueError(f"Unknown static approach: {approach_name}")
    return result.feasible, result


def run_llm_approach(approach_name, sr, substrate, agent_b, kb, mb, topo_sig, mock_llm=False):
    """Run an LLM approach: returns (admitted, plan_dict, violations, plan_shape)."""
    if mock_llm:
        # Mock: use FFD as a stand-in for Agent B
        result = _run_greedy_ffd(substrate, sr, GreedyConfig())
        plan_dict = {}
        violations = []
        if result.feasible and result.plan:
            plan_dict = {"vnf_placements": result.plan.vnf_placements}
        else:
            violations = ["structural_infeasible"]
        admitted = result.feasible
        plan_shape = _extract_plan_shape(result, sr, substrate) if result.feasible else None
        return admitted, result, plan_dict, violations, plan_shape

    # Real LLM path
    abstract_topo = build_abstract_topology(substrate)
    sr_dict = _slice_request_to_dict(sr, substrate)

    if approach_name == "Memory-off":
        plan_dict, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=None, max_retries=0,
            topology_signature=topo_sig)
    else:
        plan_dict, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=mb, max_retries=0,
            topology_signature=topo_sig)

    if not check.is_valid:
        return False, None, plan_dict, check.violations if hasattr(check, 'violations') else [], None

    # Convert LLM plan to a GreedyResult-like structure for the coordinator
    plan_result = _llm_plan_to_greedy_result(plan_dict, sr, substrate)
    admitted = plan_result is not None and plan_result.feasible
    plan_shape = _extract_plan_shape(plan_result, sr, substrate) if admitted else None
    violations = [] if admitted else ["plan_infeasible"]

    return admitted, plan_result, plan_dict, violations, plan_shape


def _slice_request_to_dict(sr, substrate):
    """Convert SliceRequest to dict for Agent B, resolving tier names."""
    g = substrate.graph
    return {
        "request_id": sr.request_id,
        "slice_type": sr.slice_type.value,
        "vnfs": [
            {
                "vnf_id": v.vnf_id,
                "vnf_type": v.vnf_type,
                "cpu_demand": v.cpu_demand,
                "ram_demand": v.ram_demand,
                "permitted_tiers": sorted({
                    g.nodes[n]["tier"] for n in v.permitted_nodes
                    if n in g.nodes
                }),
                "computational_intensity": v.computational_intensity,
                "vcr": v.vcr,
            }
            for v in sr.vnfs
        ],
        "flow_edges": [
            {
                "source_vnf": f.source_vnf,
                "target_vnf": f.target_vnf,
                "bandwidth_demand": f.bandwidth_demand,
            }
            for f in sr.flow_edges
        ],
        "qos": {
            "max_e2e_delay": sr.qos.max_e2e_delay,
            "min_throughput": sr.qos.min_throughput,
        },
    }


def _parse_abstract_domain(domain_label):
    """Map an abstract-topology domain id ('d0', 'd1', ...) to the integer
    domain_id used by the substrate/coordinator. Returns None if unparseable."""
    if isinstance(domain_label, int):
        return domain_label
    s = str(domain_label)
    if s.startswith("d"):
        s = s[1:]
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _llm_plan_to_greedy_result(plan_dict, sr, substrate):
    """Confine the concrete placer to the LLM-chosen domain per VNF (WP8 fix).

    The LLM chooses the PARTITION (a domain per VNF via ``vnf_assignments``);
    the fill then selects only a concrete node WITHIN that assigned domain,
    using the SAME plain best-fit selection (``_select_node_in_domain``) that
    Plain-ColocFB uses — so this figure isolates partition quality with the
    fill held fixed.

    If a VNF's assigned domain has no feasible node, that is a placement
    FAILURE for the LLM approach (returns None). We do NOT fall back to a free
    placement — that reintroduces the stub the diagnosis flagged.

    Routing/verification is left to the shared coordinator pipeline
    (``follow_prior`` → GreedyDomainActor → router), identical across approaches;
    this function only establishes the LLM domain (+ representative node/tier)
    that ``plan_to_summary`` reads into ``PlanSummary.suggested_domains``.
    """
    assignments = plan_dict.get("vnf_assignments", [])
    if not assignments:
        return None

    dom_of_vnf = {}
    for a in assignments:
        vid = a.get("vnf_id")
        dom = _parse_abstract_domain(a.get("domain"))
        if vid is None or dom is None:
            return None
        dom_of_vnf[vid] = dom

    g = substrate.graph
    domain_nodes: dict[int, set] = {}
    for nid, nd in g.nodes(data=True):
        domain_nodes.setdefault(nd.get("domain_id", -1), set()).add(nid)

    state = _PlacementState()
    # Largest-demand-first, matching the plain co-location fill order.
    ordered_vnfs = sorted(
        sr.vnfs, key=lambda v: (-v.cpu_demand, -v.ram_demand, v.vnf_id)
    )
    for vnf in ordered_vnfs:
        dom = dom_of_vnf.get(vnf.vnf_id)
        # Domain the LLM named must be a real substrate domain (guards the
        # coordinator's unmapped-domain index-out-of-range caveat).
        if dom is None or dom not in domain_nodes:
            return None
        nid = _select_node_in_domain(
            substrate, vnf, state, sorted(domain_nodes[dom])
        )
        if nid is None:
            return None  # assigned domain infeasible -> placement failure
        state.running_cpu[nid] = state.cpu_after(substrate, nid) - vnf.cpu_demand
        state.running_ram[nid] = state.ram_after(substrate, nid) - vnf.ram_demand
        state.vnf_placements[vnf.vnf_id] = nid
        state.cpu_allocations[vnf.vnf_id] = vnf.cpu_demand
        state.ram_allocations[vnf.vnf_id] = vnf.ram_demand
        state.resource_cost += vnf.cpu_demand + vnf.ram_demand

    plan = PlacementPlan(
        plan_id=f"{sr.request_id}_llm_partition",
        vnf_placements=state.vnf_placements,
        cpu_allocations=state.cpu_allocations,
        ram_allocations=state.ram_allocations,
        flow_routes={}, bw_allocations={},
        is_structurally_valid=True, source="llm_partition",
    )
    return GreedyResult(feasible=True, cost=state.resource_cost, plan=plan,
                        intra_bw=0.0, inter_bw=0.0,
                        resource_cost=state.resource_cost)


def _extract_plan_shape(result, sr, substrate):
    """Extract plan_shape dict for M^B recording."""
    if result is None or not result.feasible or result.plan is None:
        return None
    g = substrate.graph
    domains_used = set()
    tier_assignment = []
    for vnf in sr.vnfs:
        nid = result.plan.vnf_placements.get(vnf.vnf_id)
        if nid:
            domains_used.add(g.nodes[nid]["domain_id"])
            tier_assignment.append(g.nodes[nid]["tier"])

    strategy = "co-locate" if len(domains_used) <= 1 else "split"
    cut_points = []
    inter_domain_links = []

    if len(domains_used) > 1:
        for fe in sr.flow_edges:
            src_nid = result.plan.vnf_placements.get(fe.source_vnf)
            dst_nid = result.plan.vnf_placements.get(fe.target_vnf)
            if src_nid and dst_nid:
                if g.nodes[src_nid]["domain_id"] != g.nodes[dst_nid]["domain_id"]:
                    cut_points.append((fe.source_vnf, fe.target_vnf))
        for fk, lids in result.plan.flow_routes.items():
            for lid in lids:
                for u, v, d in g.edges(data=True):
                    if d["link_id"] == lid and g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
                        inter_domain_links.append(lid)

    return {
        "strategy": strategy,
        "tier_assignment": tier_assignment,
        "cut_points": cut_points,
        "inter_domain_links": inter_domain_links,
        "domains_used": sorted(domains_used),
    }


# ── M^B write logic ────────────────────────────────────────────────────────


def write_to_mb(mb, approach, sr, admitted, plan_dict, violations, topo_sig, plan_shape,
                committed_partition=None, diverged=None, condition_sig=None):
    """Write episode to M^B according to the approach's write rule.

    committed_partition / diverged (optional) record what the RL coordinator
    actually committed vs Agent B's suggestion — the second outcome loop.

    condition_sig (§Y.6) is the network condition AT DECISION TIME — the
    retrieval state key under a fixed topology. Callers must snapshot it before
    allocating this arrival, or every entry records a post-commit state that the
    next decision can never match."""
    if mb is None:
        return

    slice_spec = {
        "slice_type": sr.slice_type.value,
        "num_vnfs": len(sr.vnfs),
        "request_id": sr.request_id,
    }
    reward = 1.0 if admitted else -1.0
    violation_tag = violations[0] if violations else None

    # Full-M^B: selective write + importance eviction
    mb.record(
        slice_spec=slice_spec,
        plan=plan_dict,
        m_committed=0.0,
        constraints_violated=violations,
        reward=reward,
        topology_signature=topo_sig,
        plan_shape=plan_shape,
        violation_tag=violation_tag,
        committed_partition=committed_partition,
        diverged=diverged,
        condition_signature=condition_sig,
    )


# ── Instance runner ─────────────────────────────────────────────────────────


def run_instance(substrate, arrival_seed, approach_name, agent_b=None, kb=None,
                 mb=None, topo_sig=None, mock_llm=False, phase="warmup"):
    """Run one approach on one instance, return InstanceResult.

    §P frozen-M^B: M^B is written ONLY during the warm-up phase. During held-out
    (extrap) and interpolation eval the store is read-only, so the causal claim
    is transfer across topology signatures, not in-family adaptation.

    ALL approaches go through the same coordinator pipeline to equalize the gate:
      plan builder → PlanSummary (domain assignments) → coordinator (follow_prior)
      → deterministic actor (node selection) → router → admission
    """
    import copy as _copy
    from orion.actors.greedy_domain_actor import GreedyDomainActor
    from orion.mdo.coordinator import MDOConfig, MDOCoordinator

    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, ARRIVALS_PER_INSTANCE, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    # Build coordinator with deterministic actors (same for all approaches)
    actors = {d: GreedyDomainActor(d) for d in range(substrate.num_domains)}
    coord = MDOCoordinator(None, actors, MDOConfig())
    delays = {}
    g = substrate.graph
    for u, v in g.edges():
        ud, vd = g.nodes[u].get("domain_id", -1), g.nodes[v].get("domain_id", -1)
        if ud != vd and ud >= 0 and vd >= 0:
            delays[(min(ud, vd), max(ud, vd))] = min(
                delays.get((min(ud, vd), max(ud, vd)), 999.0),
                float(g[u][v]["propagation_delay"])
            )

    admitted = 0
    total = 0
    reasons = Counter()
    # §P read-only telemetry: retrieval-hit composition on held-out. If the
    # eviction ablation (Full vs FIFO) differs, this is the first place the
    # mechanism shows; if they tie, it tells us whether retrieval differentiated.
    retr = {"q": 0, "hits": 0, "pos": 0, "neg": 0}

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        total += 1
        sr = event.slice_request
        plan_dict = {}
        violations = []
        plan_shape = None

        # Step 1: Get domain assignments from the approach's plan builder
        if approach_name in STATIC_APPROACHES:
            ok_builder, builder_result = run_static_approach(approach_name, substrate, sr)
            if not ok_builder:
                reasons["structural"] += 1
                continue
            plan_summary = plan_to_summary(builder_result, sr, substrate)
        else:
            ok_builder, builder_result, plan_dict, violations, plan_shape = run_llm_approach(
                approach_name, sr, substrate, agent_b, kb, mb, topo_sig, mock_llm)
            # §P telemetry: accumulate what the agent actually retrieved from M^B.
            if mb is not None and getattr(mb, "_last_retrieval", None) is not None:
                lr = mb._last_retrieval
                retr["q"] += 1
                retr["hits"] += lr["n"]
                retr["pos"] += lr["pos"]
                retr["neg"] += lr["neg"]
            if not ok_builder or builder_result is None:
                # `violations` may be structural-checker Violation dataclasses
                # (unhashable) or strings. Reduce to hashable constraint tags for
                # both the reject-reason Counter and the M^B write (which stores
                # and later serializes them).
                v_tags = [getattr(v, "constraint", v) for v in violations]
                for vt in v_tags:
                    reasons[vt] += 1
                if approach_name in MB_APPROACHES and phase == "warmup":
                    write_to_mb(mb, approach_name, sr, False, plan_dict, v_tags,
                               topo_sig, plan_shape)
                continue
            plan_summary = plan_to_summary(builder_result, sr, substrate)

        if plan_summary is None:
            reasons["plan_conversion_fail"] += 1
            continue

        # Step 2: Route through the SAME coordinator pipeline for ALL approaches
        mdo_result = coord.resolve_arrival(
            _copy.deepcopy(substrate), sr, plan_summary, delays, mode="follow_prior")

        ok = mdo_result.admitted
        if ok:
            admitted += 1
        else:
            # Classify rejection
            if mdo_result.decision is not None:
                last = mdo_result.decision
                if last.violation:
                    v = last.violation
                    if v.actor_infeasible:
                        reasons["actor_infeasible"] += 1
                    elif v.cross_domain_infeasible:
                        reasons["cross_domain_bw"] += 1
                    elif v.c7_violated:
                        reasons["c7_delay"] += 1
                    else:
                        reasons["other_violation"] += 1
                else:
                    reasons["unknown"] += 1

        # Extract plan shape for M^B (if applicable)
        if approach_name not in STATIC_APPROACHES:
            if ok and plan_shape is None:
                plan_shape = _extract_plan_shape(builder_result, sr, substrate)
            if approach_name in MB_APPROACHES and phase == "warmup":
                v_tag = None
                if not ok and mdo_result.decision is not None:
                    last = mdo_result.decision
                    if last.violation:
                        if last.violation.cross_domain_infeasible:
                            v_tag = "C5b"
                        elif last.violation.c7_violated:
                            v_tag = "C7"
                        elif last.violation.actor_infeasible:
                            v_tag = "actor_infeasible"
                write_to_mb(mb, approach_name, sr, ok, plan_dict,
                           [v_tag] if v_tag else [], topo_sig, plan_shape)

    # §P retrieval-composition telemetry (held-out only, where the store is frozen
    # and read-only so the numbers reflect transfer, not in-family accumulation).
    if approach_name in MB_APPROACHES and phase == "extrap" and retr["q"] > 0:
        logger.info(
            "  [retr %s] held-out: mean_hits=%.2f pos=%d neg=%d over %d queries",
            approach_name, retr["hits"] / retr["q"], retr["pos"], retr["neg"], retr["q"])

    return admitted, total, dict(reasons)


# ── Main experiment loop ────────────────────────────────────────────────────




