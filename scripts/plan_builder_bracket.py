#!/usr/bin/env python3
"""Plan builder bracket: static kill recovery + dynamic depletion test.

Two stronger static plan builders vs greedy FFD:

  1. CO-LOCATION-FIRST: try placing all VNFs in one domain first.
     For each domain (sorted by residual CPU desc), check if all VNFs fit.
     If yes, run FFD constrained to that domain. Fall back to regular FFD.

  2. SCARCITY-AWARE: like FFD but node selection prefers co-locating with
     previously-placed VNFs from the same slice (minimizes cross-domain flows)
     and only considers cross-domain placement if no same-domain node fits.

Part 1: Static kill recovery per builder (fresh substrate, 5 seeds)
Part 2: Dynamic episodes (accumulating substrate with departures)
  - Watch for depletion signature in reject-reason breakdown over time
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.baselines.greedy_ffd import (
    GreedyConfig, GreedyResult, _PlacementState, _run_greedy_ffd,
    _shortest_bw_feasible_path, _link_endpoints, _required_tier,
)
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier, LinkType, PlacementPlan, SliceRequest, VNF

import torch
from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_DOMAINS = 5
INTER_DOMAIN_BW = 200.0
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
NUM_ARRIVALS = 200
SUBSTRATE_SEED = 0
ARRIVAL_SEEDS = [42, 123, 456, 789, 1001]


def build_substrate(seed=SUBSTRATE_SEED):
    rng = np.random.default_rng(seed)
    sub = generate_multi_domain_topology(
        TopologyConfig(
            num_domains=NUM_DOMAINS, nodes_per_domain=[3, 3, 3, 3, 3],
            intra_link_density=0.5, inter_domain_links=3,
            tier_distribution={
                "ran_edge": 0.10, "mec": 0.10,
                "regional_cloud": 0.40, "central_cloud": 0.40,
            },
        ), rng,
    )
    for u, v, d in sub.graph.edges(data=True):
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            d["bw_capacity"] = INTER_DOMAIN_BW
            d["bw_residual"] = INTER_DOMAIN_BW
    return sub


# ── Co-location-first builder ──────────────────────────────────────────────


def _run_colocation_ffd(substrate, slice_req, config):
    """Co-location-first: try single-domain placement before cross-domain.

    For each domain (sorted by total residual CPU desc):
      - Check if all VNFs have a feasible node of the right tier
      - If yes, run FFD constrained to that domain
    Falls back to regular FFD if no single domain works.
    """
    g = substrate.graph

    # Score each domain by total residual CPU
    domain_cpu = {}
    for nid, d in g.nodes(data=True):
        dom = d.get("domain_id", -1)
        if dom < 0:
            continue
        domain_cpu[dom] = domain_cpu.get(dom, 0.0) + float(d["cpu_residual"])

    # Try each domain, best-first
    for dom in sorted(domain_cpu.keys(), key=lambda d: -domain_cpu[d]):
        domain_nodes = {nid for nid, d in g.nodes(data=True) if d.get("domain_id") == dom}

        # Check if all VNFs can fit in this domain
        all_fit = True
        state = _PlacementState()

        ordered_vnfs = sorted(
            slice_req.vnfs,
            key=lambda f: (-f.cpu_demand, -f.ram_demand, f.vnf_id),
        )

        for vnf in ordered_vnfs:
            node_id = _select_node_in_domain(substrate, vnf, state, domain_nodes)
            if node_id is None:
                all_fit = False
                break
            state.running_cpu[node_id] = state.cpu_after(substrate, node_id) - vnf.cpu_demand
            state.running_ram[node_id] = state.ram_after(substrate, node_id) - vnf.ram_demand
            state.vnf_placements[vnf.vnf_id] = node_id
            state.cpu_allocations[vnf.vnf_id] = vnf.cpu_demand
            state.ram_allocations[vnf.vnf_id] = vnf.ram_demand
            state.resource_cost += vnf.cpu_demand + vnf.ram_demand

        if not all_fit:
            continue

        # Route flows (all intra-domain since single domain)
        route_ok = True
        for flow in slice_req.flow_edges:
            src_node = state.vnf_placements[flow.source_vnf]
            dst_node = state.vnf_placements[flow.target_vnf]
            if src_node == dst_node:
                state.flow_routes[(flow.source_vnf, flow.target_vnf)] = []
                state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = {}
                continue
            link_ids = _shortest_bw_feasible_path(
                substrate, src_node, dst_node, flow.bandwidth_demand, state
            )
            if link_ids is None:
                route_ok = False
                break
            per_link_bw = {}
            for lid in link_ids:
                u, v = _link_endpoints(substrate, lid)
                state.running_bw[lid] = state.bw_after(substrate, lid, u, v) - flow.bandwidth_demand
                per_link_bw[lid] = flow.bandwidth_demand
                state.intra_bw_cost += flow.bandwidth_demand
            state.flow_routes[(flow.source_vnf, flow.target_vnf)] = link_ids
            state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = per_link_bw

        if not route_ok:
            continue

        # Success — single-domain placement
        total_cost = (config.alpha * state.resource_cost
                      + config.gamma_intra * state.intra_bw_cost
                      + config.gamma_inter * state.inter_bw_cost)
        plan = PlacementPlan(
            plan_id=f"{slice_req.request_id}_coloc",
            vnf_placements=state.vnf_placements,
            cpu_allocations=state.cpu_allocations,
            ram_allocations=state.ram_allocations,
            flow_routes=state.flow_routes,
            bw_allocations=state.bw_allocations,
            is_structurally_valid=True, source="colocation",
        )
        return GreedyResult(feasible=True, cost=total_cost, plan=plan,
                            intra_bw=state.intra_bw_cost, inter_bw=0.0,
                            resource_cost=state.resource_cost)

    # Fallback to regular FFD
    return _run_greedy_ffd(substrate, slice_req, config)


def _select_node_in_domain(substrate, vnf, state, domain_nodes):
    """Select the best feasible node within a specific domain."""
    g = substrate.graph
    permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None
    required_tier = _required_tier(vnf, substrate)
    candidates = []

    for node_id in domain_nodes:
        d = g.nodes[node_id]
        if permitted is not None and node_id not in permitted:
            continue
        cpu_avail = state.cpu_after(substrate, node_id)
        ram_avail = state.ram_after(substrate, node_id)
        if cpu_avail < vnf.cpu_demand or ram_avail < vnf.ram_demand:
            continue
        tier_match = 1 if (required_tier and d["tier"] == required_tier) else 0
        candidates.append((-tier_match, -cpu_avail, node_id))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


# ── Scarcity-aware builder ──────────────────────────────────────────────────


def _run_scarcity_ffd(substrate, slice_req, config):
    """Scarcity-aware FFD: prefer co-locating with already-placed VNFs.

    Like FFD but the node sort key adds a co-location bonus: nodes in domains
    that already have VNFs from this slice get a preference. Cross-domain
    only when no same-domain node fits. Also prefers domains with scarce
    tiers (domains where the required tier has fewer nodes get priority, since
    those tiers will exhaust first).
    """
    g = substrate.graph
    state = _PlacementState()

    ordered_vnfs = sorted(
        slice_req.vnfs,
        key=lambda f: (-f.cpu_demand, -f.ram_demand, f.vnf_id),
    )

    placed_domains = set()

    for vnf in ordered_vnfs:
        node_id = _select_node_scarcity(substrate, vnf, state, placed_domains)
        if node_id is None:
            return GreedyResult(feasible=False, cost=float("inf"), plan=None,
                                fail_reason=f"no feasible node for VNF {vnf.vnf_id}")

        state.running_cpu[node_id] = state.cpu_after(substrate, node_id) - vnf.cpu_demand
        state.running_ram[node_id] = state.ram_after(substrate, node_id) - vnf.ram_demand
        state.vnf_placements[vnf.vnf_id] = node_id
        state.cpu_allocations[vnf.vnf_id] = vnf.cpu_demand
        state.ram_allocations[vnf.vnf_id] = vnf.ram_demand
        state.resource_cost += vnf.cpu_demand + vnf.ram_demand
        placed_domains.add(g.nodes[node_id]["domain_id"])

    # Route flows
    for flow in slice_req.flow_edges:
        src_node = state.vnf_placements[flow.source_vnf]
        dst_node = state.vnf_placements[flow.target_vnf]
        if src_node == dst_node:
            state.flow_routes[(flow.source_vnf, flow.target_vnf)] = []
            state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = {}
            continue
        link_ids = _shortest_bw_feasible_path(
            substrate, src_node, dst_node, flow.bandwidth_demand, state
        )
        if link_ids is None:
            return GreedyResult(feasible=False, cost=float("inf"), plan=None,
                                fail_reason=f"no BW-feasible path for flow {flow.source_vnf}->{flow.target_vnf}")
        per_link_bw = {}
        for lid in link_ids:
            u, v = _link_endpoints(substrate, lid)
            state.running_bw[lid] = state.bw_after(substrate, lid, u, v) - flow.bandwidth_demand
            per_link_bw[lid] = flow.bandwidth_demand
            link_type = g[u][v]["link_type"]
            if link_type == LinkType.INTER.value or link_type == LinkType.INTER:
                state.inter_bw_cost += flow.bandwidth_demand
            else:
                state.intra_bw_cost += flow.bandwidth_demand
        state.flow_routes[(flow.source_vnf, flow.target_vnf)] = link_ids
        state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = per_link_bw

    total_cost = (config.alpha * state.resource_cost
                  + config.gamma_intra * state.intra_bw_cost
                  + config.gamma_inter * state.inter_bw_cost)
    plan = PlacementPlan(
        plan_id=f"{slice_req.request_id}_scarcity",
        vnf_placements=state.vnf_placements,
        cpu_allocations=state.cpu_allocations,
        ram_allocations=state.ram_allocations,
        flow_routes=state.flow_routes,
        bw_allocations=state.bw_allocations,
        is_structurally_valid=True, source="scarcity",
    )
    return GreedyResult(feasible=True, cost=total_cost, plan=plan,
                        intra_bw=state.intra_bw_cost, inter_bw=state.inter_bw_cost,
                        resource_cost=state.resource_cost)


def _select_node_scarcity(substrate, vnf, state, placed_domains):
    """Node selection with co-location bonus + tier-scarcity awareness."""
    g = substrate.graph
    permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None
    required_tier = _required_tier(vnf, substrate)
    candidates = []

    for node_id, d in g.nodes(data=True):
        if permitted is not None and node_id not in permitted:
            continue
        cpu_avail = state.cpu_after(substrate, node_id)
        ram_avail = state.ram_after(substrate, node_id)
        if cpu_avail < vnf.cpu_demand or ram_avail < vnf.ram_demand:
            continue

        dom = d["domain_id"]
        tier_match = 1 if (required_tier and d["tier"] == required_tier) else 0
        # Co-location bonus: prefer domains already used by this slice
        coloc_bonus = 1 if dom in placed_domains else 0
        # If no VNFs placed yet, no co-location preference
        if not placed_domains:
            coloc_bonus = 0

        candidates.append((
            -tier_match,     # Must match tier first
            -coloc_bonus,    # Prefer same domain as existing VNFs
            -cpu_avail,      # Prefer most capacity
            dom,             # Tiebreak: lowest domain
            node_id,
        ))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][4]


# ── Static kill recovery ───────────────────────────────────────────────────


def run_static_bracket(arrival_seed, substrate):
    """Run all three builders on fresh substrate, count kills."""
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    cfg = GreedyConfig()
    counts = {"ffd": Counter(), "coloc": Counter(), "scarcity": Counter()}

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        sr = event.slice_request

        for name, builder in [("ffd", _run_greedy_ffd),
                              ("coloc", _run_colocation_ffd),
                              ("scarcity", _run_scarcity_ffd)]:
            result = builder(substrate, sr, cfg)
            if result.feasible:
                counts[name]["admitted"] += 1
            else:
                counts[name]["killed"] += 1

    return counts


# ── Dynamic episodes ───────────────────────────────────────────────────────


def make_plan_builder(builder_fn):
    """Wrap a builder function as a plan_builder for EpisodeRunner."""
    def plan_builder(slice_req, substrate):
        result = builder_fn(substrate, slice_req, GreedyConfig())
        if not result.feasible or result.plan is None:
            return None
        vnf_ids, required_tiers, suggested_domains = [], [], []
        for vnf in slice_req.vnfs:
            node_id = result.plan.vnf_placements[vnf.vnf_id]
            domain = int(node_id.split("n")[0][1:])
            tier_str = substrate.graph.nodes[node_id]["tier"]
            vnf_ids.append(vnf.vnf_id)
            required_tiers.append(InfrastructureTier(tier_str))
            suggested_domains.append(domain)
        return PlanSummary(
            vnf_ids=vnf_ids, required_tiers=required_tiers,
            suggested_domains=suggested_domains,
            cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
            ram_demands=[v.ram_demand for v in slice_req.vnfs],
            vcrs=[v.vcr for v in slice_req.vnfs],
            bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
        )
    return plan_builder


def build_delays(sub):
    delays = {}
    for u, v in sub.graph.edges():
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            key = (min(u_dom, v_dom), max(u_dom, v_dom))
            delays.setdefault(key, 8.0)
    return delays


def run_dynamic(arrival_seed, sub_template, delays, builder_fn, builder_name):
    """Run a full dynamic episode with one builder."""
    sub = copy.deepcopy(sub_template)
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    actors = {}
    for d in range(NUM_DOMAINS):
        torch.manual_seed(arrival_seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))

    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
    plan_builder = make_plan_builder(builder_fn)
    runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode="follow_prior")

    # Per-slice reject reasons
    reasons = Counter()
    for r in ep.mdo_results:
        if r.admitted:
            continue
        last = r.retry_history.attempts[-1] if r.retry_history.attempts else None
        if last is None or last.violation is None:
            reasons["unknown"] += 1
            continue
        v = last.violation
        if v.actor_infeasible:
            reasons["actor_infeasible"] += 1
        elif v.cross_domain_infeasible:
            reasons["cross_domain_bw"] += 1
        elif v.c7_violated:
            reasons["c7_delay"] += 1
        else:
            reasons["other"] += 1

    return ep.stats, reasons


def main():
    sub = build_substrate(SUBSTRATE_SEED)
    delays = build_delays(sub)

    logger.info("=" * 90)
    logger.info("PLAN BUILDER BRACKET — static kill recovery + dynamic depletion test")
    logger.info("  Substrate: 5 domains x 3 nodes, seed=%d", SUBSTRATE_SEED)
    logger.info("  Arrivals: %d per seed, seeds: %s", NUM_ARRIVALS, ARRIVAL_SEEDS)
    logger.info("=" * 90)

    # ── Part 1: Static kill recovery ────────────────────────────────────────

    logger.info("")
    logger.info("PART 1: STATIC KILL RECOVERY (fresh substrate)")
    logger.info("-" * 60)

    agg = {"ffd": Counter(), "coloc": Counter(), "scarcity": Counter()}
    per_seed = []

    for seed in ARRIVAL_SEEDS:
        counts = run_static_bracket(seed, sub)
        per_seed.append(counts)
        for name in agg:
            for k, v in counts[name].items():
                agg[name][k] += v

    for name in ["ffd", "coloc", "scarcity"]:
        total = agg[name]["admitted"] + agg[name]["killed"]
        adm = agg[name]["admitted"]
        killed = agg[name]["killed"]
        per = [s[name]["admitted"] for s in per_seed]
        kill_rate = 100 * killed / total if total > 0 else 0
        adm_rate = 100 * adm / total if total > 0 else 0
        logger.info("  %-12s: admitted=%d/%d (%.1f%%)  killed=%d (%.1f%%)  per-seed: %s",
                    name, adm, total, adm_rate, killed, kill_rate, per)

    # Kill recovery
    ffd_kills = agg["ffd"]["killed"]
    coloc_recovery = ffd_kills - agg["coloc"]["killed"]
    scarcity_recovery = ffd_kills - agg["scarcity"]["killed"]
    logger.info("")
    logger.info("  FFD kills:        %d", ffd_kills)
    logger.info("  Co-loc recovers:  %d (%.1f%% of FFD kills)",
                coloc_recovery, 100 * coloc_recovery / ffd_kills if ffd_kills > 0 else 0)
    logger.info("  Scarcity recovers: %d (%.1f%% of FFD kills)",
                scarcity_recovery, 100 * scarcity_recovery / ffd_kills if ffd_kills > 0 else 0)

    # ── Part 2: Dynamic episodes ────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("PART 2: DYNAMIC EPISODES (with departures)")
    logger.info("=" * 90)

    builders = [
        ("ffd", _run_greedy_ffd),
        ("coloc", _run_colocation_ffd),
        ("scarcity", _run_scarcity_ffd),
    ]

    dyn_results = {name: [] for name, _ in builders}
    dyn_reasons = {name: Counter() for name, _ in builders}
    dyn_structural = {name: [] for name, _ in builders}

    for seed in ARRIVAL_SEEDS:
        logger.info("--- Seed %d ---", seed)
        for name, builder_fn in builders:
            t0 = time.time()
            stats, reasons = run_dynamic(seed, sub, delays, builder_fn, name)
            elapsed = time.time() - t0
            pct = 100 * stats.admitted / stats.total_arrivals
            dyn_results[name].append(pct)
            dyn_structural[name].append(stats.rejected_structural)
            for k, v in reasons.items():
                dyn_reasons[name][k] += v
            logger.info("  %-12s: %3d/%d (%.1f%%)  structural=%d  time=%.1fs",
                        name, stats.admitted, stats.total_arrivals, pct,
                        stats.rejected_structural, elapsed)

    logger.info("")
    logger.info("DYNAMIC AGGREGATE (%d seeds):", len(ARRIVAL_SEEDS))
    for name, _ in builders:
        pcts = dyn_results[name]
        structs = dyn_structural[name]
        logger.info("  %-12s: %.1f%% +/- %.1f%%  structural: %s  reasons: %s",
                    name, np.mean(pcts), np.std(pcts), structs,
                    dict(dyn_reasons[name]))

    # ── Verdict ─────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("VERDICT")
    logger.info("=" * 90)

    ffd_dyn = np.mean(dyn_results["ffd"])
    coloc_dyn = np.mean(dyn_results["coloc"])
    scarcity_dyn = np.mean(dyn_results["scarcity"])

    logger.info("  Static kill recovery (of %d FFD kills):", ffd_kills)
    logger.info("    Co-location-first: %d recovered (%.1f%%)",
                coloc_recovery, 100 * coloc_recovery / ffd_kills if ffd_kills else 0)
    logger.info("    Scarcity-aware:    %d recovered (%.1f%%)",
                scarcity_recovery, 100 * scarcity_recovery / ffd_kills if ffd_kills else 0)
    logger.info("")
    logger.info("  Dynamic admission:")
    logger.info("    FFD:               %.1f%%", ffd_dyn)
    logger.info("    Co-location-first: %.1f%%  (%+.1f pp vs FFD)", coloc_dyn, coloc_dyn - ffd_dyn)
    logger.info("    Scarcity-aware:    %.1f%%  (%+.1f pp vs FFD)", scarcity_dyn, scarcity_dyn - ffd_dyn)

    if coloc_dyn < ffd_dyn:
        logger.info("")
        logger.info("  CO-LOCATION SELF-DESTRUCTS DYNAMICALLY (as predicted).")
        logger.info("  Static recovery does not survive depletion.")
    elif coloc_dyn > ffd_dyn + 3:
        logger.info("")
        logger.info("  CO-LOCATION SURVIVES DYNAMICALLY (+%.1f pp).", coloc_dyn - ffd_dyn)
    else:
        logger.info("")
        logger.info("  CO-LOCATION IS MARGINAL DYNAMICALLY (+%.1f pp).", coloc_dyn - ffd_dyn)

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
