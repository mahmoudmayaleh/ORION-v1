#!/usr/bin/env python3
"""Regenerate greedy and actor baselines under the real cross-domain router.

Every admission number prior to this script was computed with cross-domain
flows silently dropped. This is the first honest measurement.

Usage:
    python scripts/baseline_under_real_router.py --seed 0
"""

from __future__ import annotations

import argparse
import json
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
from orion.baselines.greedy_ffd import greedy_place_on_substrate, GreedyConfig, _run_greedy_ffd
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline")

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


def run_greedy_raw(substrate, seed):
    """Greedy FFD with no commit gate — raw upper bound."""
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    substrate.reset()
    ap.reset()
    total = admitted = 0
    while ap.has_next():
        ev = ap.next_event()
        if ev.event_type == EventType.DEPARTURE:
            p = substrate._active_slices.get(ev.request_id)
            if p:
                substrate.deallocate(p[0], p[1])
            continue
        total += 1
        r = greedy_place_on_substrate(substrate, ev.slice_request)
        if r.feasible:
            admitted += 1
    return total, admitted


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
        vnf_ids=vnf_ids,
        required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


def run_episode_with_router(substrate, seed, actors, mode="follow_prior"):
    """Run a full episode through the real cross-domain router."""
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    delays = build_inter_domain_delays(substrate)
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=3))
    runner = EpisodeRunner(substrate, ap, coord, delays, plan_builder=_greedy_plan_builder)
    runner.reset()
    return runner.run_episode(mdo_mode=mode)


def analyze_rejects(ep):
    reasons = Counter()
    for r in ep.mdo_results:
        if not r.admitted:
            for a in r.retry_history.attempts:
                if a.violation:
                    v = a.violation
                    if v.actor_infeasible:
                        reasons["actor_infeasible"] += 1
                    if v.cross_domain_infeasible:
                        reasons["cross_domain_infeasible"] += 1
                    if v.c7_violated:
                        reasons["c7_delay"] += 1
                    if v.c9_violated:
                        reasons["c9_hops"] += 1
                    if v.c5b_violated:
                        reasons["c5b_aggregate"] += 1
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    logger.info("=" * 60)
    logger.info("BASELINE UNDER REAL CROSS-DOMAIN ROUTER")
    logger.info("=" * 60)

    # 1. Greedy raw (no gate, no router)
    substrate = build_substrate(seed)
    t0 = time.time()
    total, greedy_raw = run_greedy_raw(substrate, seed)
    logger.info(f"Greedy (raw): {greedy_raw}/{total} ({100*greedy_raw/total:.1f}%) "
                f"in {time.time()-t0:.1f}s")

    # 2. Greedy-aligned actors through the real router (follow_prior)
    substrate = build_substrate(seed)
    actors = {}
    for d in range(3):
        torch.manual_seed(seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))

    t0 = time.time()
    ep = run_episode_with_router(substrate, seed, actors, mode="follow_prior")
    elapsed = time.time() - t0

    admitted = ep.stats.admitted
    total_arr = ep.stats.total_arrivals
    structural = ep.stats.rejected_structural
    mdo_reject = ep.stats.rejected_by_mdo

    logger.info(f"Untrained actors (follow_prior, real router): "
                f"{admitted}/{total_arr} ({100*admitted/total_arr:.1f}%) in {elapsed:.1f}s")
    logger.info(f"  Structural rejects: {structural}")
    logger.info(f"  MDO rejects: {mdo_reject}")

    # Cross-domain stats
    cross_admitted = sum(1 for r in ep.mdo_results if r.admitted and r.cross_domain_routes)
    cross_total = sum(1 for r in ep.mdo_results
                      if r.retry_history.attempts and
                      any(a.violation and a.violation.cross_domain_infeasible
                          for a in r.retry_history.attempts))
    logger.info(f"  Admitted with cross-domain routes: {cross_admitted}")
    logger.info(f"  Rejected with cross-domain infeasible: {cross_total}")

    # Reject breakdown
    reasons = analyze_rejects(ep)
    logger.info("  Reject breakdown (per-attempt):")
    for k, v in reasons.most_common():
        logger.info(f"    {k}: {v}")

    # Hard penalty fires
    logger.info(f"  Hard penalty fires: {ep.stats.hard_penalty_fires}")

    # 3. Random partition (to force cross-domain diversity)
    substrate = build_substrate(seed)
    actors_rand = {}
    for d in range(3):
        torch.manual_seed(seed + d + 100)
        actors_rand[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))

    t0 = time.time()
    ep_rand = run_episode_with_router(substrate, seed, actors_rand, mode="random")
    elapsed = time.time() - t0

    logger.info(f"Untrained actors (random partition, real router): "
                f"{ep_rand.stats.admitted}/{ep_rand.stats.total_arrivals} "
                f"({100*ep_rand.stats.admitted/ep_rand.stats.total_arrivals:.1f}%) in {elapsed:.1f}s")
    cross_admitted_rand = sum(1 for r in ep_rand.mdo_results if r.admitted and r.cross_domain_routes)
    logger.info(f"  Admitted with cross-domain routes: {cross_admitted_rand}")
    reasons_rand = analyze_rejects(ep_rand)
    logger.info("  Reject breakdown (per-attempt):")
    for k, v in reasons_rand.most_common():
        logger.info(f"    {k}: {v}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY — First honest numbers")
    logger.info("=" * 60)
    logger.info(f"  Greedy (raw, no gate):          {greedy_raw}/{total} ({100*greedy_raw/total:.1f}%)")
    logger.info(f"  Untrained (follow_prior):       {admitted}/{total_arr} ({100*admitted/total_arr:.1f}%)")
    logger.info(f"  Untrained (random partition):   {ep_rand.stats.admitted}/{ep_rand.stats.total_arrivals} "
                f"({100*ep_rand.stats.admitted/ep_rand.stats.total_arrivals:.1f}%)")
    logger.info(f"  Cross-domain routes admitted:    follow_prior={cross_admitted}, random={cross_admitted_rand}")


if __name__ == "__main__":
    main()
