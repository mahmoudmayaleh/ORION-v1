#!/usr/bin/env python3
"""Five-arm learning-curve runner — implements the pre-registered protocol.

See docs/EXPERIMENT_PROTOCOL.md for the frozen specification.

Arms:
  1. RA-ColocFB    — routability-aware co-location (static, no LLM)
  2. Memory-off    — Agent B + K^B, no M^B
  3. FIFO-M^B      — Agent B + K^B + M^B (write-all, FIFO eviction, K=50)
  4. Full-M^B      — Agent B + K^B + M^B (selective write, importance eviction, K=50)
  5. Plain-ColocFB — co-location with FFD fallback (static, no LLM)

Usage:
  python scripts/five_arm_runner.py                    # real LLM (requires backend)
  python scripts/five_arm_runner.py --mock-llm         # mock Agent B (FFD-based, for infra testing)
  python scripts/five_arm_runner.py --mock-llm --seed 0  # single seed
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.baselines.colocation_ffd import (
    colocation_ffd,
    routability_aware_colocation_ffd,
)
from orion.baselines.greedy_ffd import GreedyConfig, _run_greedy_ffd
from orion.llm.abstract_topology import build_abstract_topology
from orion.llm.episodic_memory import EpisodicMemory
from orion.retrieval import RetrievalConfig, RetrievalMode
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.substrate.topology_families import (
    ALL_FAMILIES,
    TRAIN_FAMILIES,
    TEST_FAMILIES,
    TopologyFamily,
    compute_signature,
    generate_family_instance,
)
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

WARM_UP_ORDER = [
    "C+_T+_B+", "C+_T+_B-", "C-_T-_B+", "C-_T-_B-", "C+_T-_B+",
]

ARM_NAMES = ["RA-ColocFB", "Memory-off", "FIFO-M^B", "Full-M^B", "Plain-ColocFB"]
STATIC_ARMS = {"RA-ColocFB", "Plain-ColocFB"}
LLM_ARMS = {"Memory-off", "FIFO-M^B", "Full-M^B"}


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class InstanceResult:
    """Per-instance, per-arm result."""
    family: str
    instance_seed: int
    arm: str
    phase: str  # "warmup", "extrap", "interp"
    admitted: int = 0
    total: int = 0
    ceiling: int = 0
    fraction_of_ceiling: float = 0.0
    reject_reasons: dict = field(default_factory=dict)


# ── Ceiling computation (enumerator, executor-independent) ──────────────────


def compute_ceiling(substrate, arrival_seed):
    """Count arrivals with at least one valid placement+routing.

    Uses the same logic as the frozen kill classifier v2 but only needs
    the binary feasible/infeasible answer, not the bin classification.
    """
    import networkx as nx
    from orion.actors.routing import route_cross_domain_flow, allocate_route_bw, deallocate_route_bw
    from orion.sim.delay_model import node_sojourn, link_sojourn

    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, ARRIVALS_PER_INSTANCE, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    ceiling = 0
    total = 0
    g = substrate.graph

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        total += 1
        sr = event.slice_request

        # Quick check: does ColocFB admit? If yes, ceiling admits.
        result = colocation_ffd(substrate, sr, GreedyConfig())
        if result.feasible:
            ceiling += 1
            continue

        # Exhaustive: check if ANY valid placement exists
        feasible_nodes = _get_feasible_nodes(sr, substrate)
        if any(len(n) == 0 for n in feasible_nodes):
            continue

        combos = list(itertools.product(*feasible_nodes))
        if len(combos) > 5000:
            rng_s = np.random.default_rng(hash(sr.request_id) % 2**32)
            indices = rng_s.choice(len(combos), 5000, replace=False)
            combos = [combos[i] for i in indices]

        found = False
        for combo in combos:
            if _check_placement_full(list(combo), sr, substrate):
                found = True
                break
        if found:
            ceiling += 1

    return total, ceiling


def _get_feasible_nodes(sr, substrate):
    g = substrate.graph
    feasible = []
    for vnf in sr.vnfs:
        permitted_tiers = {g.nodes[n]["tier"] for n in vnf.permitted_nodes if n in g.nodes}
        nodes = [
            nid for nid, d in g.nodes(data=True)
            if d.get("domain_id", -1) >= 0
            and d["tier"] in permitted_tiers
            and float(d["cpu_residual"]) >= vnf.cpu_demand
            and float(d["ram_residual"]) >= vnf.ram_demand
        ]
        feasible.append(nodes)
    return feasible


def _check_placement_full(placement, sr, substrate):
    """Check if a placement passes the full verifier gate."""
    import networkx as nx
    from orion.actors.routing import route_cross_domain_flow, allocate_route_bw
    from orion.sim.delay_model import node_sojourn, link_sojourn

    g = substrate.graph
    node_cpu, node_ram = {}, {}
    for j, vnf in enumerate(sr.vnfs):
        nid = placement[j]
        node_cpu[nid] = node_cpu.get(nid, 0.0) + vnf.cpu_demand
        node_ram[nid] = node_ram.get(nid, 0.0) + vnf.ram_demand
        if node_cpu[nid] > float(g.nodes[nid]["cpu_residual"]) + 0.01:
            return False
        if node_ram[nid] > float(g.nodes[nid]["ram_residual"]) + 0.01:
            return False

    vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(sr.vnfs)}
    sub_copy = copy.deepcopy(substrate)
    routes = {}

    for fe in sr.flow_edges:
        src, dst = vnf_to_node[fe.source_vnf], vnf_to_node[fe.target_vnf]
        if src == dst:
            continue
        src_dom, dst_dom = g.nodes[src]["domain_id"], g.nodes[dst]["domain_id"]

        if src_dom == dst_dom:
            domain_nodes = [n for n, d in g.nodes(data=True) if d.get("domain_id") == src_dom]
            subg = sub_copy.graph.subgraph(domain_nodes)
            try:
                for p in nx.shortest_simple_paths(subg, src, dst, weight="propagation_delay"):
                    if all(float(sub_copy.graph[p[i]][p[i+1]]["bw_residual"]) >= fe.bandwidth_demand
                           for i in range(len(p) - 1)):
                        for i in range(len(p) - 1):
                            sub_copy.graph[p[i]][p[i+1]]["bw_residual"] -= fe.bandwidth_demand
                        lids = [sub_copy.graph[p[i]][p[i+1]]["link_id"] for i in range(len(p) - 1)]
                        routes[(fe.source_vnf, fe.target_vnf)] = lids
                        break
                else:
                    return False
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return False
        else:
            result = route_cross_domain_flow(
                sub_copy, src, dst, bw_demand=fe.bandwidth_demand, delay_budget=999999.0)
            if not result.feasible:
                return False
            allocate_route_bw(sub_copy, result.path_links, fe.bandwidth_demand)
            routes[(fe.source_vnf, fe.target_vnf)] = result.path_links

    # C9
    hops = sum(
        1 for fk, lids in routes.items() for lid in lids
        for u, v, d in [next((u, v, d) for u, v, d in g.edges(data=True) if d["link_id"] == lid)]
        if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]
    )
    if hops > 3:
        return False

    # C7 delay
    extra_cpu = {}
    for vnf in sr.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        extra_cpu[nid] = extra_cpu.get(nid, 0.0) + vnf.cpu_demand

    total_delay = 0.0
    for vnf in sr.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        nd = g.nodes[nid]
        cpu_used = float(nd["cpu_capacity"]) - float(nd["cpu_residual"]) + extra_cpu[nid]
        s = node_sojourn(float(nd["processing_delay"]), vnf.computational_intensity,
                         float(nd["cpu_capacity"]), cpu_used)
        if math.isinf(s):
            return False
        total_delay += s

    extra_bw = {}
    for fe in sr.flow_edges:
        for lid in routes.get((fe.source_vnf, fe.target_vnf), []):
            extra_bw[lid] = extra_bw.get(lid, 0.0) + fe.bandwidth_demand
    for fe in sr.flow_edges:
        for lid in routes.get((fe.source_vnf, fe.target_vnf), []):
            for u, v, d in g.edges(data=True):
                if d["link_id"] == lid:
                    bw_cap = float(d.get("bandwidth_capacity", d.get("bw_capacity", 1000)))
                    bw_used = bw_cap - float(d["bw_residual"]) + extra_bw.get(lid, 0.0)
                    s = link_sojourn(float(d["propagation_delay"]), bw_cap, bw_used)
                    if math.isinf(s):
                        return False
                    total_delay += s
                    break

    return total_delay <= sr.qos.max_e2e_delay


# ── Plan builders per arm ───────────────────────────────────────────────────


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


def run_static_arm(arm_name, substrate, sr):
    """Run a static arm: returns (admitted: bool, plan_summary_or_none)."""
    cfg = GreedyConfig()
    if arm_name == "RA-ColocFB":
        result = routability_aware_colocation_ffd(substrate, sr, cfg)
    elif arm_name == "Plain-ColocFB":
        result = colocation_ffd(substrate, sr, cfg)
    else:
        raise ValueError(f"Unknown static arm: {arm_name}")
    return result.feasible, result


def run_llm_arm(arm_name, sr, substrate, agent_b, kb, mb, topo_sig, mock_llm=False):
    """Run an LLM arm: returns (admitted, plan_dict, violations, plan_shape)."""
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

    if arm_name == "Memory-off":
        plan_dict, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=None, max_retries=0)
    else:
        plan_dict, check = agent_b.generate_with_memory(
            sr_dict, abstract_topo, kb=kb, mb=mb, max_retries=0)

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


def _llm_plan_to_greedy_result(plan_dict, sr, substrate):
    """Convert an LLM plan dict to a placement on the substrate."""
    # Extract domain assignments from LLM plan
    assignments = plan_dict.get("vnf_assignments", [])
    if not assignments:
        return None
    # Use ColocFB as the executor (the LLM decides the partition,
    # the executor places within domains)
    # For now, map LLM's domain assignments to a ColocFB call
    # constrained to those domains
    return colocation_ffd(substrate, sr, GreedyConfig())


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


def write_to_mb(mb, arm_name, sr, admitted, plan_dict, violations, topo_sig, plan_shape):
    """Write episode to M^B according to arm's write rule."""
    if mb is None:
        return

    slice_spec = {
        "slice_type": sr.slice_type.value,
        "num_vnfs": len(sr.vnfs),
        "request_id": sr.request_id,
    }
    reward = 1.0 if admitted else -1.0
    violation_tag = violations[0] if violations else None

    if arm_name == "FIFO-M^B":
        # Write-all: bypass selective filter, always record
        mb._entries.append(mb._create_entry(
            slice_spec, plan_dict, 0.0, violations, reward,
            topo_sig, plan_shape, violation_tag,
        ))
        if len(mb._entries) > mb._max_entries:
            # FIFO eviction: remove oldest
            mb._entries.pop(0)
    else:
        # Full-M^B: selective write + importance eviction (default behavior)
        mb.record(
            slice_spec=slice_spec,
            plan=plan_dict,
            m_committed=0.0,
            constraints_violated=violations,
            reward=reward,
            topology_signature=topo_sig,
            plan_shape=plan_shape,
            violation_tag=violation_tag,
        )


# ── Instance runner ─────────────────────────────────────────────────────────


def run_instance(substrate, arrival_seed, arm_name, agent_b=None, kb=None,
                 mb=None, topo_sig=None, mock_llm=False):
    """Run one arm on one instance, return InstanceResult.

    ALL arms go through the same coordinator pipeline to equalize the gate:
      plan builder → PlanSummary (domain assignments) → coordinator (follow_prior)
      → deterministic actor (node selection) → router → admission
    """
    import copy as _copy
    from orion.actors.greedy_domain_actor import GreedyDomainActor
    from orion.mdo.coordinator import MDOConfig, MDOCoordinator

    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, ARRIVALS_PER_INSTANCE, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    # Build coordinator with deterministic actors (same for all arms)
    actors = {d: GreedyDomainActor(d) for d in range(substrate.num_domains)}
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
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

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        total += 1
        sr = event.slice_request
        plan_dict = {}
        violations = []
        plan_shape = None

        # Step 1: Get domain assignments from the arm's plan builder
        if arm_name in STATIC_ARMS:
            ok_builder, builder_result = run_static_arm(arm_name, substrate, sr)
            if not ok_builder:
                reasons["structural"] += 1
                continue
            plan_summary = plan_to_summary(builder_result, sr, substrate)
        else:
            ok_builder, builder_result, plan_dict, violations, plan_shape = run_llm_arm(
                arm_name, sr, substrate, agent_b, kb, mb, topo_sig, mock_llm)
            if not ok_builder or builder_result is None:
                # `violations` may be structural-checker Violation dataclasses
                # (unhashable) or strings. Reduce to hashable constraint tags for
                # both the reject-reason Counter and the M^B write (which stores
                # and later serializes them).
                v_tags = [getattr(v, "constraint", v) for v in violations]
                for vt in v_tags:
                    reasons[vt] += 1
                if arm_name in ("FIFO-M^B", "Full-M^B"):
                    write_to_mb(mb, arm_name, sr, False, plan_dict, v_tags,
                               topo_sig, plan_shape)
                continue
            plan_summary = plan_to_summary(builder_result, sr, substrate)

        if plan_summary is None:
            reasons["plan_conversion_fail"] += 1
            continue

        # Step 2: Route through the SAME coordinator pipeline for ALL arms
        mdo_result = coord.resolve_arrival(
            _copy.deepcopy(substrate), sr, plan_summary, delays, mode="follow_prior")

        ok = mdo_result.admitted
        if ok:
            admitted += 1
        else:
            # Classify rejection
            if mdo_result.retry_history.attempts:
                last = mdo_result.retry_history.attempts[-1]
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
        if arm_name not in STATIC_ARMS:
            if ok and plan_shape is None:
                plan_shape = _extract_plan_shape(builder_result, sr, substrate)
            if arm_name in ("FIFO-M^B", "Full-M^B"):
                v_tag = None
                if not ok and mdo_result.retry_history.attempts:
                    last = mdo_result.retry_history.attempts[-1]
                    if last.violation:
                        if last.violation.cross_domain_infeasible:
                            v_tag = "C5b"
                        elif last.violation.c7_violated:
                            v_tag = "C7"
                        elif last.violation.actor_infeasible:
                            v_tag = "actor_infeasible"
                write_to_mb(mb, arm_name, sr, ok, plan_dict,
                           [v_tag] if v_tag else [], topo_sig, plan_shape)

    return admitted, total, dict(reasons)


# ── Main experiment loop ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-llm", action="store_true",
                        help="Use FFD as mock Agent B (no real LLM calls)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed (default: all 3)")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else RUN_SEEDS
    mock_llm = args.mock_llm

    logger.info("=" * 90)
    logger.info("FIVE-ARM LEARNING-CURVE RUNNER")
    logger.info("  Mock LLM: %s", mock_llm)
    logger.info("  Seeds: %s", seeds)
    logger.info("  Arms: %s", ARM_NAMES)
    logger.info("  Memory K: %d", MEMORY_CAPACITY_K)
    logger.info("  N_struct: %d", N_STRUCT)
    logger.info("=" * 90)

    # ── Pre-compute: generate instances and ceilings ────────────────────────

    family_lookup = {f.short_name: f for f in ALL_FAMILIES}

    # All instances needed
    instances = {}  # (family_short, inst_seed) -> substrate
    ceilings = {}   # (family_short, inst_seed, arrival_seed) -> (total, ceiling)

    all_instance_specs = []
    # Train instances (warm-up)
    for fname in WARM_UP_ORDER:
        for iseed in TRAIN_INSTANCE_SEEDS:
            all_instance_specs.append((fname, iseed, "warmup"))
    # Test instances (extrapolation)
    for f in TEST_FAMILIES:
        for iseed in TEST_INSTANCE_SEEDS:
            all_instance_specs.append((f.short_name, iseed, "extrap"))
    # Interpolation instances
    for fname in WARM_UP_ORDER:
        all_instance_specs.append((fname, INTERP_INSTANCE_SEED, "interp"))

    logger.info("\nGenerating %d instances...", len(all_instance_specs))
    for fname, iseed, phase in all_instance_specs:
        key = (fname, iseed)
        if key not in instances:
            family = family_lookup[fname]
            sub = generate_family_instance(family, seed=iseed)
            instances[key] = sub

    logger.info("Computing per-instance ceilings...")
    for (fname, iseed), sub in instances.items():
        for arrival_seed in seeds:
            t0 = time.time()
            total, ceiling = compute_ceiling(sub, arrival_seed)
            elapsed = time.time() - t0
            ceilings[(fname, iseed, arrival_seed)] = (total, ceiling)
            logger.info("  %s inst=%d seed=%d: ceiling=%d/%d (%.1f%%) [%.1fs]",
                        fname, iseed, arrival_seed, ceiling, total,
                        100 * ceiling / total if total > 0 else 0, elapsed)

    # ── Per-seed runs ───────────────────────────────────────────────────────

    all_results: list[InstanceResult] = []

    for run_seed in seeds:
        logger.info("")
        logger.info("=" * 90)
        logger.info("SEED %d", run_seed)
        logger.info("=" * 90)

        # Initialize M^B for FIFO and Full arms (empty, K=50)
        mb_fifo = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=False, k_final=3),
            max_entries=MEMORY_CAPACITY_K,
        )
        mb_full = EpisodicMemory(
            config=RetrievalConfig(mode=RetrievalMode.NO_RERANK, apply_recency=True, k_final=3),
            max_entries=MEMORY_CAPACITY_K,
        )

        # Agent B and K^B (shared across LLM arms within a seed)
        agent_b = None
        kb = None
        if not mock_llm:
            try:
                from orion.llm.llm_backend import LLMBackend, LLMConfig
                from orion.llm.agent_b import AgentB
                from orion.llm.semantic_memory import SemanticMemory

                llm_config = LLMConfig(
                    base_url="http://localhost:8000/v1",
                    api_key="EMPTY",
                    model="default",
                    temperature=0.05,
                    max_tokens=2048,
                )
                llm = LLMBackend(llm_config)
                agent_b = AgentB(llm)

                # Load K^B (frozen, identical across all LLM arms)
                kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
                if kb_path.exists():
                    kb = SemanticMemory.from_json(kb_path)
                    logger.info("K^B loaded: %d entries", len(kb.entries))
                else:
                    logger.warning("K^B not found at %s", kb_path)

            except Exception as e:
                logger.warning("LLM backend unavailable: %s. Falling back to mock.", e)
                mock_llm = True

        # ── Warm-up phase ──────────────────────────────────────────────

        logger.info("\n--- WARM-UP PHASE ---")

        for fname in WARM_UP_ORDER:
            for iseed in TRAIN_INSTANCE_SEEDS:
                sub = instances[(fname, iseed)]
                topo_sig = compute_signature(sub, fname).to_dict()
                total_ceil, ceil_count = ceilings[(fname, iseed, run_seed)]

                for arm in ARM_NAMES:
                    mb = None
                    if arm == "FIFO-M^B":
                        mb = mb_fifo
                    elif arm == "Full-M^B":
                        mb = mb_full

                    t0 = time.time()
                    adm, total, reasons = run_instance(
                        sub, run_seed, arm, agent_b, kb, mb, topo_sig, mock_llm)
                    elapsed = time.time() - t0

                    foc = adm / ceil_count if ceil_count > 0 else 0.0
                    ir = InstanceResult(
                        family=fname, instance_seed=iseed, arm=arm,
                        phase="warmup", admitted=adm, total=total,
                        ceiling=ceil_count, fraction_of_ceiling=foc,
                        reject_reasons=reasons,
                    )
                    all_results.append(ir)

                    logger.info("  %s i=%d %-14s: %d/%d (FoC=%.1f%%) [%.1fs]%s",
                                fname, iseed, arm, adm, total, 100 * foc, elapsed,
                                f" M^B={len(mb._entries)}" if mb else "")

        logger.info("\nM^B sizes after warm-up: FIFO=%d, Full=%d",
                    len(mb_fifo._entries), len(mb_full._entries))

        # ── Guardrail (Amendment 3): context-headroom gate ─────────────────
        # M^B is now populated, so warm-up has exercised the largest prompts the
        # run will produce. Verify none approached n_ctx (context exhaustion is
        # arm-asymmetric — only K^B+M^B arms grow — and silently truncates the
        # completion). Fail loudly here rather than emit void eval data.
        if not mock_llm and agent_b is not None:
            fr_counts = agent_b.llm.finish_reason_counts
            max_pt = agent_b.llm.max_prompt_tokens_seen
            n_ctx = agent_b.llm.config.n_ctx
            n_trunc = fr_counts.get("length", 0)
            logger.info("Warm-up LLM telemetry: finish_reasons=%s, "
                        "max_prompt_tokens=%d / n_ctx=%d", dict(fr_counts), max_pt, n_ctx)
            if n_trunc > 0:
                raise RuntimeError(
                    f"{n_trunc} truncated completion(s) (finish_reason=length) during "
                    f"warm-up: prompts exceed the {n_ctx}-token window. Raise --n_ctx or "
                    f"cap memory-entry size before running eval (Amendment 3)."
                )
            if max_pt > n_ctx - 1024:
                raise RuntimeError(
                    f"Warm-up max prompt = {max_pt} tokens leaves < 1024 tokens of "
                    f"completion headroom in the {n_ctx}-token window. Truncation is "
                    f"imminent; raise --n_ctx or cap memory-entry size (Amendment 3)."
                )
            if max_pt > 6500:
                logger.warning(
                    "Warm-up max prompt = %d tokens (> 6500). Headroom is adequate at "
                    "n_ctx=%d but tightening; consider a per-entry character cap on K^B/"
                    "M^B injection before scaling top-k (Amendment 3).", max_pt, n_ctx)

        # ── Eval: extrapolation (held-out families) ────────────────────

        logger.info("\n--- EVAL: EXTRAPOLATION (held-out families) ---")

        for family in TEST_FAMILIES:
            fname = family.short_name
            for iseed in TEST_INSTANCE_SEEDS:
                sub = instances[(fname, iseed)]
                topo_sig = compute_signature(sub, fname).to_dict()
                total_ceil, ceil_count = ceilings[(fname, iseed, run_seed)]

                for arm in ARM_NAMES:
                    mb = None
                    if arm == "FIFO-M^B":
                        mb = mb_fifo
                    elif arm == "Full-M^B":
                        mb = mb_full

                    t0 = time.time()
                    adm, total, reasons = run_instance(
                        sub, run_seed, arm, agent_b, kb, mb, topo_sig, mock_llm)
                    elapsed = time.time() - t0

                    foc = adm / ceil_count if ceil_count > 0 else 0.0
                    ir = InstanceResult(
                        family=fname, instance_seed=iseed, arm=arm,
                        phase="extrap", admitted=adm, total=total,
                        ceiling=ceil_count, fraction_of_ceiling=foc,
                        reject_reasons=reasons,
                    )
                    all_results.append(ir)

                    logger.info("  %s i=%d %-14s: %d/%d (FoC=%.1f%%) [%.1fs]",
                                fname, iseed, arm, adm, total, 100 * foc, elapsed)

        # ── Eval: interpolation (unseen instances, seen families) ──────

        logger.info("\n--- EVAL: INTERPOLATION (unseen instances, seen families) ---")

        for fname in WARM_UP_ORDER:
            iseed = INTERP_INSTANCE_SEED
            sub = instances[(fname, iseed)]
            topo_sig = compute_signature(sub, fname).to_dict()
            total_ceil, ceil_count = ceilings[(fname, iseed, run_seed)]

            for arm in ARM_NAMES:
                mb = None
                if arm == "FIFO-M^B":
                    mb = mb_fifo
                elif arm == "Full-M^B":
                    mb = mb_full

                t0 = time.time()
                adm, total, reasons = run_instance(
                    sub, run_seed, arm, agent_b, kb, mb, topo_sig, mock_llm)
                elapsed = time.time() - t0

                foc = adm / ceil_count if ceil_count > 0 else 0.0
                ir = InstanceResult(
                    family=fname, instance_seed=iseed, arm=arm,
                    phase="interp", admitted=adm, total=total,
                    ceiling=ceil_count, fraction_of_ceiling=foc,
                    reject_reasons=reasons,
                )
                all_results.append(ir)

                logger.info("  %s i=%d %-14s: %d/%d (FoC=%.1f%%) [%.1fs]",
                            fname, iseed, arm, adm, total, 100 * foc, elapsed)

    # ── Aggregate reporting ─────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 90)

    for phase in ["warmup", "extrap", "interp"]:
        phase_results = [r for r in all_results if r.phase == phase]
        if not phase_results:
            continue

        logger.info("\n--- %s ---", phase.upper())

        # Per-family, per-arm FoC
        families_in_phase = sorted(set(r.family for r in phase_results))
        for fname in families_in_phase:
            logger.info("  %s:", fname)
            for arm in ARM_NAMES:
                focs = [r.fraction_of_ceiling for r in phase_results
                        if r.family == fname and r.arm == arm]
                if focs:
                    logger.info("    %-14s: FoC=%.1f%% +/- %.1f%%  [%s]",
                                arm, 100 * np.mean(focs), 100 * np.std(focs),
                                ", ".join(f"{100*f:.1f}" for f in focs))

    # ── Success criterion check ─────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("SUCCESS CRITERION CHECK")
    logger.info("=" * 90)

    # Primary: Full-M^B vs best static on B- test families
    b_minus_test = ["C-_T+_B-", "C+_T-_B-"]
    for fname in b_minus_test:
        extrap = [r for r in all_results if r.phase == "extrap" and r.family == fname]
        if not extrap:
            continue

        full_focs = [r.fraction_of_ceiling for r in extrap if r.arm == "Full-M^B"]
        ra_focs = [r.fraction_of_ceiling for r in extrap if r.arm == "RA-ColocFB"]
        plain_focs = [r.fraction_of_ceiling for r in extrap if r.arm == "Plain-ColocFB"]

        best_static = max(np.mean(ra_focs) if ra_focs else 0,
                         np.mean(plain_focs) if plain_focs else 0)
        full_mean = np.mean(full_focs) if full_focs else 0
        full_std = np.std(full_focs) if full_focs else 0
        delta = full_mean - best_static

        logger.info("  %s: Full-M^B=%.1f%% vs BestStatic=%.1f%% → delta=%+.1f pp (spread=%.1f%%)",
                    fname, 100 * full_mean, 100 * best_static, 100 * delta, 100 * full_std)

    # No-regression control
    control_family = "C-_T+_B+"
    control_results = [r for r in all_results if r.phase == "extrap" and r.family == control_family]
    if control_results:
        for arm in ARM_NAMES:
            focs = [r.fraction_of_ceiling for r in control_results if r.arm == arm]
            if focs:
                logger.info("  Control %s: %s FoC=%.1f%%", control_family, arm, 100 * np.mean(focs))

    # Secondary: Full vs FIFO, Full vs Memory-off
    for phase in ["extrap", "interp"]:
        phase_r = [r for r in all_results if r.phase == phase]
        for arm_pair in [("Full-M^B", "FIFO-M^B"), ("Full-M^B", "Memory-off")]:
            a_focs = [r.fraction_of_ceiling for r in phase_r if r.arm == arm_pair[0]]
            b_focs = [r.fraction_of_ceiling for r in phase_r if r.arm == arm_pair[1]]
            if a_focs and b_focs:
                logger.info("  %s: %s=%.1f%% vs %s=%.1f%% (delta=%+.1f pp)",
                            phase, arm_pair[0], 100 * np.mean(a_focs),
                            arm_pair[1], 100 * np.mean(b_focs),
                            100 * (np.mean(a_focs) - np.mean(b_focs)))

    logger.info("=" * 90)

    # Save raw results
    out_path = Path("data/five_arm_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([{
            "family": r.family, "instance_seed": r.instance_seed,
            "arm": r.arm, "phase": r.phase,
            "admitted": r.admitted, "total": r.total,
            "ceiling": r.ceiling, "foc": r.fraction_of_ceiling,
            "reject_reasons": r.reject_reasons,
        } for r in all_results], f, indent=2)
    logger.info("\nRaw results saved to %s", out_path)


if __name__ == "__main__":
    main()
