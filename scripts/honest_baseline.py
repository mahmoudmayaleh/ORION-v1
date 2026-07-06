#!/usr/bin/env python3
"""Honest baseline: greedy through the real router + untrained actors.

Produces the first valid comparison table. All numbers in the same
feasibility universe (physical-edge cross-domain routing).

Usage:
    python scripts/honest_baseline.py --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.actors.routing import route_cross_domain_flow, allocate_route_bw, deallocate_route_bw
from orion.baselines.greedy_ffd import greedy_place_on_substrate, _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("honest_baseline")

NUM_ARRIVALS = 2000
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
HIDDEN_DIM = 64


def build_substrate(seed):
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(num_domains=3, nodes_per_domain=[8, 10, 12],
                       intra_link_density=0.4, inter_domain_links=4), rng)


def build_inter_domain_delays(substrate):
    delays = {}
    for u, v, d in substrate.graph.edges(data=True):
        sd = substrate.graph.nodes[u]["domain_id"]
        dd = substrate.graph.nodes[v]["domain_id"]
        if sd != dd:
            delays[(sd, dd)] = min(delays.get((sd, dd), float("inf")), d["propagation_delay"])
    return delays


def run_greedy_gated(substrate, seed, max_inter_hops=3):
    """Greedy FFD with cross-domain flows routed through the real router.

    For each greedy admission, extract the partition, identify cross-domain
    flows, route them on the full graph, check C7 (E2E delay) and C9 (hops).
    Only admit if all gates pass. This is greedy in the router's universe.
    """
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    substrate.reset()
    ap.reset()

    total = admitted_raw = admitted_gated = 0
    gate_rejects = Counter()

    while ap.has_next():
        ev = ap.next_event()
        if ev.event_type == EventType.DEPARTURE:
            p = substrate._active_slices.get(ev.request_id)
            if p:
                substrate.deallocate(p[0], p[1])
            continue

        total += 1
        sr = ev.slice_request
        result = _run_greedy_ffd(substrate, sr, GreedyConfig())
        if not result.feasible or result.plan is None:
            continue

        admitted_raw += 1

        # Extract greedy's partition
        partition = []
        for vnf in sr.vnfs:
            node_id = result.plan.vnf_placements[vnf.vnf_id]
            domain = int(node_id.split("n")[0][1:])
            partition.append(domain)

        # Identify cross-domain flows and route them
        vnf_ids = [v.vnf_id for v in sr.vnfs]
        vnf_to_idx = {vid: i for i, vid in enumerate(vnf_ids)}

        cross_failed = False
        cross_routes = {}
        cross_bw_alloc = {}
        cross_delay = 0.0
        cross_hops = 0

        for fe in sr.flow_edges:
            src_idx = vnf_to_idx[fe.source_vnf]
            dst_idx = vnf_to_idx[fe.target_vnf]
            if partition[src_idx] == partition[dst_idx]:
                continue

            src_node = result.plan.vnf_placements[fe.source_vnf]
            dst_node = result.plan.vnf_placements[fe.target_vnf]

            route_result = route_cross_domain_flow(
                substrate, src_node, dst_node,
                bw_demand=fe.bandwidth_demand,
                delay_budget=sr.qos.max_e2e_delay - cross_delay,
            )

            if not route_result.feasible:
                cross_failed = True
                # Rollback any provisional BW
                for fk, bw in cross_bw_alloc.items():
                    deallocate_route_bw(substrate, cross_routes[fk], bw)
                break

            fk = (fe.source_vnf, fe.target_vnf)
            cross_routes[fk] = route_result.path_links
            cross_bw_alloc[fk] = fe.bandwidth_demand
            cross_delay += route_result.propagation_delay
            allocate_route_bw(substrate, route_result.path_links, fe.bandwidth_demand)

            # Count inter-domain hops
            g = substrate.graph
            for lid in route_result.path_links:
                for u, v, d in g.edges(data=True):
                    if d["link_id"] == lid:
                        if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
                            cross_hops += 1
                        break

        if cross_failed:
            gate_rejects["cross_domain_bw"] += 1
            continue

        # Compute total E2E including greedy's intra-domain delay
        intra_delay = 0.0
        for (src_vnf, dst_vnf), link_ids in result.plan.flow_routes.items():
            for lid in link_ids:
                for u, v, d in substrate.graph.edges(data=True):
                    if d["link_id"] == lid:
                        intra_delay += d["propagation_delay"]
                        break

        e2e = intra_delay + cross_delay
        if e2e > sr.qos.max_e2e_delay:
            # Rollback cross-domain BW
            for fk, bw in cross_bw_alloc.items():
                deallocate_route_bw(substrate, cross_routes[fk], bw)
            gate_rejects["c7_delay"] += 1
            continue

        if cross_hops > max_inter_hops:
            for fk, bw in cross_bw_alloc.items():
                deallocate_route_bw(substrate, cross_routes[fk], bw)
            gate_rejects["c9_hops"] += 1
            continue

        # Rollback provisional cross-domain BW (allocate() handles definitive)
        for fk, bw in cross_bw_alloc.items():
            deallocate_route_bw(substrate, cross_routes[fk], bw)

        # All gates pass — allocate the greedy plan (intra-domain routes)
        substrate.allocate(result.plan, sr)
        admitted_gated += 1

    return total, admitted_raw, admitted_gated, gate_rejects


def _greedy_plan_builder(slice_req, substrate):
    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
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


def run_untrained(substrate, seed):
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    delays = build_inter_domain_delays(substrate)
    actors = {}
    for d in range(3):
        torch.manual_seed(seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=3))
    runner = EpisodeRunner(substrate, ap, coord, delays, plan_builder=_greedy_plan_builder)
    runner.reset()
    return runner.run_episode(mdo_mode="follow_prior")


def per_slice_rejects(ep):
    """Per-slice ultimate reject reason (last attempt's violation for rejected slices)."""
    reasons = Counter()
    for r in ep.mdo_results:
        if r.admitted:
            continue
        # Use the last attempt's violation as the ultimate reason
        last = r.retry_history.attempts[-1] if r.retry_history.attempts else None
        if last is None or last.violation is None:
            reasons["unknown"] += 1
            continue
        v = last.violation
        if v.actor_infeasible:
            reasons["actor_infeasible"] += 1
        elif v.cross_domain_infeasible:
            reasons["cross_domain_infeasible"] += 1
        elif v.c7_violated:
            reasons["c7_delay"] += 1
        elif v.c9_violated:
            reasons["c9_hops"] += 1
        else:
            reasons["unknown"] += 1
    return reasons


def per_attempt_rejects_overlap_check(ep):
    """Check if any single attempt has BOTH actor_infeasible and cross_domain_infeasible."""
    overlap = 0
    total_attempts = 0
    for r in ep.mdo_results:
        if r.admitted:
            continue
        for a in r.retry_history.attempts:
            if a.violation is None:
                continue
            total_attempts += 1
            v = a.violation
            if v.actor_infeasible and v.cross_domain_infeasible:
                overlap += 1
    return overlap, total_attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    logger.info("=" * 60)
    logger.info("HONEST BASELINE — all numbers in the router's universe")
    logger.info("=" * 60)

    # 1. Gated greedy through the real router
    substrate = build_substrate(seed)
    t0 = time.time()
    total, greedy_raw, greedy_gated, gate_rejects = run_greedy_gated(substrate, seed)
    logger.info(f"Greedy (raw, no gate): {greedy_raw}/{total} ({100*greedy_raw/total:.1f}%)")
    logger.info(f"Greedy (gated, real router): {greedy_gated}/{total} "
                f"({100*greedy_gated/total:.1f}%) in {time.time()-t0:.1f}s")
    logger.info(f"  Gate rejects: {dict(gate_rejects)}")
    logger.info(f"  Gate rejection rate on raw: "
                f"{100*(greedy_raw-greedy_gated)/max(greedy_raw,1):.1f}%")

    # 2. Untrained actors (follow_prior) through real router
    substrate = build_substrate(seed)
    t0 = time.time()
    ep = run_untrained(substrate, seed)
    elapsed = time.time() - t0
    admitted = ep.stats.admitted
    total_arr = ep.stats.total_arrivals

    logger.info(f"Untrained actors (follow_prior): {admitted}/{total_arr} "
                f"({100*admitted/total_arr:.1f}%) in {elapsed:.1f}s")

    # Per-slice ultimate reject reasons
    slice_rejects = per_slice_rejects(ep)
    logger.info("  Per-SLICE ultimate reject reasons:")
    total_rejected_slices = sum(slice_rejects.values())
    for k, v in slice_rejects.most_common():
        logger.info(f"    {k}: {v} ({100*v/total_rejected_slices:.1f}% of rejects)")

    # Overlap check
    overlap, total_attempts = per_attempt_rejects_overlap_check(ep)
    logger.info(f"  Double-count check: {overlap} attempts with BOTH actor_infeasible "
                f"AND cross_domain_infeasible out of {total_attempts} failed attempts")
    if overlap == 0:
        logger.info("  CONFIRMED: no overlap, double-count fix landed")
    else:
        logger.info(f"  WARNING: {overlap} overlapping attempts — double-count still present!")

    # Cross-domain stats
    cross_admitted = sum(1 for r in ep.mdo_results if r.admitted and r.cross_domain_routes)
    logger.info(f"  Admitted with cross-domain routes: {cross_admitted}/{admitted}")
    logger.info(f"  Hard penalty fires: {ep.stats.hard_penalty_fires}")
    logger.info(f"  Structural rejects: {ep.stats.rejected_structural}")

    # Summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALID COMPARISON TABLE (same feasibility universe)")
    logger.info("=" * 60)
    logger.info(f"  Greedy (gated, real router):   {greedy_gated}/{total} "
                f"({100*greedy_gated/total:.1f}%) — TRUE CEILING")
    logger.info(f"  Untrained (follow_prior):      {admitted}/{total_arr} "
                f"({100*admitted/total_arr:.1f}%) — "
                f"{100*admitted/max(greedy_gated,1):.0f}% of gated greedy")
    logger.info(f"  Cross-domain routes admitted:   {cross_admitted}/{admitted}")
    logger.info("")
    logger.info(f"  (Raw greedy 46.2% is INVALID — different feasibility universe, no C5b/C7/C9)")


if __name__ == "__main__":
    main()
