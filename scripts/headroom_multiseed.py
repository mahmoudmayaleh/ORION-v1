#!/usr/bin/env python3
"""Multi-seed headroom confirmation: optimal vs random partition, post-fix.

Re-confirms the partition headroom on a CLEAN substrate, multiple seeds,
with per-seed and aggregate statistics. No BC. No uncontrolled variables.

Design:
  - ONE substrate configuration (5 domains x 3 nodes, 10% edge/MEC)
  - STATIC snapshot per seed (no dynamic resource accumulation)
  - Each slice evaluated independently against the same snapshot
  - 5 seeds for arrival process, 1 fixed substrate seed
  - Reports mean +/- std for all key metrics

Also runs a DYNAMIC version (full episode with departures) for each seed
to verify static headroom translates to real-world improvement.
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.actors.routing import allocate_route_bw, route_cross_domain_flow
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration (FIXED across all seeds) ──────────────────────────────────

NUM_DOMAINS = 5
INTER_DOMAIN_BW = 200.0
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
NUM_ARRIVALS = 200
SUBSTRATE_SEED = 0
ARRIVAL_SEEDS = [42, 123, 456, 789, 1001]
MAX_PARTITIONS_PER_SLICE = 2000


def build_substrate(seed=SUBSTRATE_SEED):
    rng = np.random.default_rng(seed)
    sub = generate_multi_domain_topology(
        TopologyConfig(
            num_domains=NUM_DOMAINS,
            nodes_per_domain=[3, 3, 3, 3, 3],
            intra_link_density=0.5,
            inter_domain_links=3,
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


def build_delays(sub):
    delays = {}
    for u, v in sub.graph.edges():
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            key = (min(u_dom, v_dom), max(u_dom, v_dom))
            delays.setdefault(key, 8.0)
    return delays


def greedy_plan_builder(slice_req, substrate):
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


def get_feasible_domains_per_vnf(plan, substrate):
    g = substrate.graph
    feasible_per_vnf = []
    for k in range(plan.num_vnfs):
        tier = plan.required_tiers[k]
        cpu_need = plan.cpu_demands[k]
        ram_need = plan.ram_demands[k]
        feasible_doms = set()
        for nid, d in g.nodes(data=True):
            if d.get("domain_id", -1) < 0:
                continue
            if d["tier"] != tier.value:
                continue
            if float(d["cpu_residual"]) >= cpu_need and float(d["ram_residual"]) >= ram_need:
                feasible_doms.add(d["domain_id"])
        feasible_per_vnf.append(sorted(feasible_doms))
    return feasible_per_vnf


def try_partition_via_coordinator(partition, plan, slice_req, substrate, actors, delays):
    sub_copy = copy.deepcopy(substrate)
    plan_copy = PlanSummary(
        vnf_ids=plan.vnf_ids,
        required_tiers=plan.required_tiers,
        suggested_domains=partition,
        cpu_demands=plan.cpu_demands,
        ram_demands=plan.ram_demands,
        vcrs=plan.vcrs,
        bw_demands=plan.bw_demands,
    )
    coord = MDOCoordinator(
        None, actors,
        MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
    )
    result = coord.resolve_arrival(
        sub_copy, slice_req, plan_copy, delays, mode="follow_prior",
    )
    viols = 0
    if result.retry_history.attempts:
        v = result.retry_history.attempts[-1].violation
        if v and v.has_violation:
            viols = 1
    return result.admitted, result.total_cost, result.e2e_delay, viols


@dataclass
class SeedResult:
    seed: int = 0
    total_arrivals: int = 0
    plan_failed: int = 0
    plan_survived: int = 0
    optimal_admitted: int = 0
    random_admitted: int = 0
    greedy_admitted: int = 0
    optimal_only: int = 0
    headroom_pp: float = 0.0
    elapsed_s: float = 0.0


def run_static_seed(arrival_seed: int, sub: SubstrateNetwork, delays: dict,
                    actors: dict) -> SeedResult:
    """Run the static partition comparison for one arrival seed."""
    t0 = time.time()
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    sub_snapshot = copy.deepcopy(sub)
    result = SeedResult(seed=arrival_seed, total_arrivals=NUM_ARRIVALS)

    arrival_count = 0
    for event in ap.events:
        if event.event_type == "departure" or event.slice_request is None:
            continue
        arrival_count += 1
        sr = event.slice_request

        plan = greedy_plan_builder(sr, sub_snapshot)
        if plan is None:
            result.plan_failed += 1
            continue

        feasible_doms = get_feasible_domains_per_vnf(plan, sub_snapshot)
        if any(len(f) == 0 for f in feasible_doms):
            result.plan_failed += 1
            continue

        result.plan_survived += 1

        # ── Optimal: enumerate all feasible partitions ──
        all_partitions = list(itertools.product(*feasible_doms))
        if len(all_partitions) > MAX_PARTITIONS_PER_SLICE:
            rng_sample = np.random.default_rng(hash(sr.request_id) % 2**32 + 999)
            indices = rng_sample.choice(len(all_partitions), size=MAX_PARTITIONS_PER_SLICE, replace=False)
            all_partitions = [all_partitions[i] for i in indices]

        best_admitted = False
        best_cost = float("inf")
        for part in all_partitions:
            part_list = list(part)
            admitted, cost, delay, viols = try_partition_via_coordinator(
                part_list, plan, sr, sub_snapshot, actors, delays,
            )
            if admitted:
                if not best_admitted or cost < best_cost:
                    best_admitted = True
                    best_cost = cost

        if best_admitted:
            result.optimal_admitted += 1

        # ── Random: single random partition ──
        rng_rand = np.random.default_rng(hash(sr.request_id) % 2**32)
        rand_part = [int(rng_rand.choice(f)) for f in feasible_doms]
        admitted_r, _, _, _ = try_partition_via_coordinator(
            rand_part, plan, sr, sub_snapshot, actors, delays,
        )
        if admitted_r:
            result.random_admitted += 1

        # ── Greedy: follow the plan builder's suggestion ──
        greedy_part = plan.suggested_domains
        admitted_g, _, _, _ = try_partition_via_coordinator(
            greedy_part, plan, sr, sub_snapshot, actors, delays,
        )
        if admitted_g:
            result.greedy_admitted += 1

        if best_admitted and not admitted_r:
            result.optimal_only += 1

    # Headroom
    if result.plan_survived > 0:
        opt_pct = 100 * result.optimal_admitted / result.plan_survived
        rand_pct = 100 * result.random_admitted / result.plan_survived
        result.headroom_pp = opt_pct - rand_pct
    result.elapsed_s = time.time() - t0
    return result


def run_dynamic_seed(arrival_seed: int, sub_template: SubstrateNetwork,
                     delays: dict, mode: str) -> tuple[int, int, int]:
    """Run a full dynamic episode (with departures) in the given mode.

    mode: "follow_prior" (greedy partition), "random", or "optimal" (not feasible dynamically)
    Returns: (total_arrivals, admitted, structural_rejects)
    """
    sub = copy.deepcopy(sub_template)
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    actors = {}
    for d in range(NUM_DOMAINS):
        torch.manual_seed(arrival_seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))

    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1))
    runner = EpisodeRunner(sub, ap, coord, delays,
                           plan_builder=greedy_plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode=mode)
    return ep.stats.total_arrivals, ep.stats.admitted, ep.stats.rejected_structural


def main():
    sub = build_substrate(SUBSTRATE_SEED)
    delays = build_delays(sub)
    g = sub.graph

    logger.info("=" * 90)
    logger.info("MULTI-SEED HEADROOM CONFIRMATION — post-fix, clean substrate")
    logger.info("  Substrate: 5 domains x 3 nodes, 10%% edge/10%% MEC, seed=%d", SUBSTRATE_SEED)
    logger.info("  Arrivals: %d per seed, lambda=%.1f, mu=%.3f", NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE)
    logger.info("  Arrival seeds: %s", ARRIVAL_SEEDS)
    logger.info("  Max partitions per slice: %d", MAX_PARTITIONS_PER_SLICE)
    logger.info("=" * 90)

    # Log substrate topology
    for dom in range(NUM_DOMAINS):
        tiers = {}
        cpu_total = 0
        for nid, d in g.nodes(data=True):
            if d.get("domain_id") == dom:
                t = d["tier"]
                tiers[t] = tiers.get(t, 0) + 1
                cpu_total += float(d["cpu_capacity"])
        tier_str = ", ".join(f"{t}:{c}" for t, c in sorted(tiers.items()))
        logger.info("  Domain %d: %s  (%.0f CPU total)", dom, tier_str, cpu_total)

    # Build actors (shared across seeds for consistency)
    actors = {}
    for d in range(NUM_DOMAINS):
        torch.manual_seed(d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))

    # ── PART 1: Static headroom (per-slice, no resource accumulation) ──────

    logger.info("")
    logger.info("=" * 90)
    logger.info("PART 1: STATIC HEADROOM (per-slice, independent evaluation)")
    logger.info("=" * 90)

    static_results: list[SeedResult] = []
    for seed in ARRIVAL_SEEDS:
        logger.info("")
        logger.info("--- Seed %d ---", seed)
        sr = run_static_seed(seed, sub, delays, actors)
        static_results.append(sr)
        logger.info("  Plan failed: %d/%d (%.1f%%)",
                     sr.plan_failed, NUM_ARRIVALS, 100 * sr.plan_failed / NUM_ARRIVALS)
        logger.info("  Survived: %d", sr.plan_survived)
        opt_pct = 100 * sr.optimal_admitted / max(sr.plan_survived, 1)
        rand_pct = 100 * sr.random_admitted / max(sr.plan_survived, 1)
        greedy_pct = 100 * sr.greedy_admitted / max(sr.plan_survived, 1)
        logger.info("  Optimal:  %d/%d (%.1f%%)", sr.optimal_admitted, sr.plan_survived, opt_pct)
        logger.info("  Random:   %d/%d (%.1f%%)", sr.random_admitted, sr.plan_survived, rand_pct)
        logger.info("  Greedy:   %d/%d (%.1f%%)", sr.greedy_admitted, sr.plan_survived, greedy_pct)
        logger.info("  Headroom: %.1f pp (%d slices optimal-only)", sr.headroom_pp, sr.optimal_only)
        logger.info("  Time: %.1fs", sr.elapsed_s)

    # Aggregate
    logger.info("")
    logger.info("=" * 90)
    logger.info("STATIC AGGREGATE (%d seeds)", len(ARRIVAL_SEEDS))
    logger.info("=" * 90)

    plan_fail_pcts = [100 * r.plan_failed / NUM_ARRIVALS for r in static_results]
    opt_pcts = [100 * r.optimal_admitted / max(r.plan_survived, 1) for r in static_results]
    rand_pcts = [100 * r.random_admitted / max(r.plan_survived, 1) for r in static_results]
    greedy_pcts = [100 * r.greedy_admitted / max(r.plan_survived, 1) for r in static_results]
    headrooms = [r.headroom_pp for r in static_results]
    opt_overall = [100 * r.optimal_admitted / NUM_ARRIVALS for r in static_results]
    rand_overall = [100 * r.random_admitted / NUM_ARRIVALS for r in static_results]
    greedy_overall = [100 * r.greedy_admitted / NUM_ARRIVALS for r in static_results]

    logger.info("  Plan-builder kill rate:  %.1f%% +/- %.1f%%",
                np.mean(plan_fail_pcts), np.std(plan_fail_pcts))
    logger.info("")
    logger.info("  Of survivors:")
    logger.info("    Optimal admission:   %.1f%% +/- %.1f%%", np.mean(opt_pcts), np.std(opt_pcts))
    logger.info("    Greedy admission:    %.1f%% +/- %.1f%%", np.mean(greedy_pcts), np.std(greedy_pcts))
    logger.info("    Random admission:    %.1f%% +/- %.1f%%", np.mean(rand_pcts), np.std(rand_pcts))
    logger.info("    HEADROOM (opt-rand): %.1f pp +/- %.1f pp", np.mean(headrooms), np.std(headrooms))
    logger.info("")
    logger.info("  Overall (incl plan failures):")
    logger.info("    Optimal admission:   %.1f%% +/- %.1f%%", np.mean(opt_overall), np.std(opt_overall))
    logger.info("    Greedy admission:    %.1f%% +/- %.1f%%", np.mean(greedy_overall), np.std(greedy_overall))
    logger.info("    Random admission:    %.1f%% +/- %.1f%%", np.mean(rand_overall), np.std(rand_overall))

    # ── PART 2: Dynamic episodes (with departures) ─────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("PART 2: DYNAMIC EPISODES (with departures, resource accumulation)")
    logger.info("=" * 90)

    dyn_greedy = []
    dyn_random = []
    for seed in ARRIVAL_SEEDS:
        logger.info("--- Seed %d ---", seed)
        t0 = time.time()
        total_g, admitted_g, struct_g = run_dynamic_seed(seed, sub, delays, "follow_prior")
        total_r, admitted_r, struct_r = run_dynamic_seed(seed, sub, delays, "random")
        elapsed = time.time() - t0
        dyn_greedy.append(100 * admitted_g / total_g)
        dyn_random.append(100 * admitted_r / total_r)
        logger.info("  Greedy partition: %d/%d (%.1f%%), structural=%d",
                     admitted_g, total_g, 100 * admitted_g / total_g, struct_g)
        logger.info("  Random partition: %d/%d (%.1f%%), structural=%d",
                     admitted_r, total_r, 100 * admitted_r / total_r, struct_r)
        logger.info("  Time: %.1fs", elapsed)

    logger.info("")
    logger.info("DYNAMIC AGGREGATE (%d seeds):", len(ARRIVAL_SEEDS))
    logger.info("  Greedy partition: %.1f%% +/- %.1f%%", np.mean(dyn_greedy), np.std(dyn_greedy))
    logger.info("  Random partition: %.1f%% +/- %.1f%%", np.mean(dyn_random), np.std(dyn_random))
    logger.info("  Dynamic headroom: %.1f pp +/- %.1f pp",
                np.mean(dyn_greedy) - np.mean(dyn_random),
                np.sqrt(np.std(dyn_greedy)**2 + np.std(dyn_random)**2))

    # ── VERDICT ────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("VERDICT")
    logger.info("=" * 90)

    mean_headroom = np.mean(headrooms)
    std_headroom = np.std(headrooms)
    min_headroom = min(headrooms)

    if min_headroom > 5.0:
        logger.info("  HEADROOM CONFIRMED: %.1f +/- %.1f pp (min=%.1f, all seeds > 5 pp)",
                    mean_headroom, std_headroom, min_headroom)
        logger.info("  Partition decision matters. Proceed to two-layer bracket.")
    elif mean_headroom > 5.0:
        logger.info("  HEADROOM LIKELY: %.1f +/- %.1f pp (min=%.1f, some seeds marginal)",
                    mean_headroom, std_headroom, min_headroom)
    elif mean_headroom > 0:
        logger.info("  SMALL HEADROOM: %.1f +/- %.1f pp (min=%.1f). Partition matters modestly.",
                    mean_headroom, std_headroom, min_headroom)
    else:
        logger.info("  NO HEADROOM: %.1f +/- %.1f pp. Random is near-optimal.",
                    mean_headroom, std_headroom)

    # Three-level bracket
    mean_greedy_pct = np.mean(greedy_pcts)
    mean_opt_pct = np.mean(opt_pcts)
    mean_rand_pct = np.mean(rand_pcts)
    logger.info("")
    logger.info("  THREE-LEVEL BRACKET (of survivors):")
    logger.info("    Random partition:    %.1f%%", mean_rand_pct)
    logger.info("    Greedy partition:    %.1f%%", mean_greedy_pct)
    logger.info("    Optimal partition:   %.1f%%", mean_opt_pct)
    logger.info("    Greedy over random:  %.1f pp", mean_greedy_pct - mean_rand_pct)
    logger.info("    Optimal over greedy: %.1f pp", mean_opt_pct - mean_greedy_pct)
    logger.info("    Optimal over random: %.1f pp", mean_opt_pct - mean_rand_pct)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
