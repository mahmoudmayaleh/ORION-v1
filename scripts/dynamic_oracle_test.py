#!/usr/bin/env python3
"""Decisive test: does myopic per-slice optimal survive dynamic accumulation?

For each arrival on the REAL accumulating substrate:
  1. Plan via greedy FFD on current state
  2. Enumerate feasible partitions, test each on a deep copy
  3. Pick the best-admitting partition (lowest cost)
  4. Commit it on the real substrate (allocate, track, depart)

If myopic-optimal beats random dynamically: myopic partition intelligence
transfers, and the MDO's job is to approximate it.

If myopic-optimal loses to random (as greedy did): the job is sequential,
and the bandit formulation is the bottleneck.

Also runs:
  - Random partition (dynamic baseline — currently the strongest)
  - Greedy partition (follow_prior — known to self-destruct)
  - Load-balance partition (per-VNF: pick domain with most residual CPU)

All four on same substrate seed, same arrival seeds, multi-seed.
"""

from __future__ import annotations

import copy
import itertools
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
from orion.actors.routing import allocate_route_bw, deallocate_route_bw, route_cross_domain_flow
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier, PlacementPlan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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


def try_partition(partition, plan, slice_req, substrate, actors, delays):
    """Test a partition on a deep copy. Returns (admitted, cost, mdo_result)."""
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
    return result.admitted, result.total_cost, result


def build_placement_plan(slice_req, mdo_result):
    """Build a PlacementPlan from MDOResult (same logic as EpisodeRunner)."""
    vnf_placements = {}
    cpu_allocations = {}
    ram_allocations = {}
    flow_routes = {}
    bw_allocations = {}
    vnf_by_id = {v.vnf_id: v for v in slice_req.vnfs}

    for _domain_id, response in mdo_result.domain_responses.items():
        for vnf_id, node_id in response.placements.items():
            vnf_placements[vnf_id] = node_id
            vnf = vnf_by_id[vnf_id]
            cpu_allocations[vnf_id] = vnf.cpu_demand
            ram_allocations[vnf_id] = vnf.ram_demand
        for flow_key, route_link_ids in response.routes.items():
            flow_routes[flow_key] = route_link_ids
            per_link_bw_value = response.bw_allocated.get(flow_key, 0.0)
            bw_allocations[flow_key] = {
                link_id: per_link_bw_value for link_id in route_link_ids
            }

    for flow_key, route_link_ids in mdo_result.cross_domain_routes.items():
        flow_routes[flow_key] = route_link_ids
        bw = mdo_result.cross_domain_bw.get(flow_key, 0.0)
        bw_allocations[flow_key] = {link_id: bw for link_id in route_link_ids}

    if any(v.vnf_id not in vnf_placements for v in slice_req.vnfs):
        return None

    return PlacementPlan(
        plan_id=f"{slice_req.request_id}_oracle",
        vnf_placements=vnf_placements,
        cpu_allocations=cpu_allocations,
        ram_allocations=ram_allocations,
        flow_routes=flow_routes,
        bw_allocations=bw_allocations,
        is_structurally_valid=True,
        source="oracle",
    )


def pick_load_balance_partition(plan, substrate):
    """Per-VNF: pick the feasible domain with the most residual CPU."""
    g = substrate.graph
    partition = []
    for k in range(plan.num_vnfs):
        tier = plan.required_tiers[k]
        cpu_need = plan.cpu_demands[k]
        ram_need = plan.ram_demands[k]

        best_dom = None
        best_cpu = -1.0
        for nid, d in g.nodes(data=True):
            if d.get("domain_id", -1) < 0:
                continue
            if d["tier"] != tier.value:
                continue
            if float(d["cpu_residual"]) >= cpu_need and float(d["ram_residual"]) >= ram_need:
                dom = d["domain_id"]
                cpu_res = float(d["cpu_residual"])
                if cpu_res > best_cpu:
                    best_cpu = cpu_res
                    best_dom = dom

        if best_dom is None:
            return None
        partition.append(best_dom)
    return partition


def run_dynamic_policy(arrival_seed, sub_template, delays, policy_name):
    """Run one dynamic episode with a given partition policy.

    policy_name: "oracle", "random", "greedy", "load_balance"
    """
    sub = copy.deepcopy(sub_template)
    sub.reset()

    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    ap.reset()

    actors = {}
    for d in range(NUM_DOMAINS):
        torch.manual_seed(arrival_seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))

    active_slices = {}
    total = admitted = structural = no_feasible = 0
    reject_reasons = Counter()

    while ap.has_next():
        event = ap.next_event()

        if event.event_type == EventType.DEPARTURE:
            entry = active_slices.pop(event.request_id, None)
            if entry:
                sub.deallocate(entry[0], entry[1])
            continue

        total += 1
        sr = event.slice_request

        plan = greedy_plan_builder(sr, sub)
        if plan is None:
            structural += 1
            reject_reasons["structural"] += 1
            continue

        feasible_doms = get_feasible_domains_per_vnf(plan, sub)
        if any(len(f) == 0 for f in feasible_doms):
            structural += 1
            reject_reasons["no_feasible_domain"] += 1
            continue

        # ── Select partition based on policy ──
        if policy_name == "oracle":
            all_partitions = list(itertools.product(*feasible_doms))
            if len(all_partitions) > MAX_PARTITIONS_PER_SLICE:
                rng_sample = np.random.default_rng(
                    hash(sr.request_id) % 2**32 + 999
                )
                indices = rng_sample.choice(
                    len(all_partitions), size=MAX_PARTITIONS_PER_SLICE, replace=False
                )
                all_partitions = [all_partitions[i] for i in indices]

            best_partition = None
            best_cost = float("inf")
            for part in all_partitions:
                part_list = list(part)
                adm, cost, _ = try_partition(
                    part_list, plan, sr, sub, actors, delays
                )
                if adm and cost < best_cost:
                    best_partition = part_list
                    best_cost = cost

            if best_partition is None:
                no_feasible += 1
                reject_reasons["no_admitting_partition"] += 1
                continue
            chosen_partition = best_partition

        elif policy_name == "random":
            rng_rand = np.random.default_rng(
                hash(sr.request_id) % 2**32 + arrival_seed
            )
            chosen_partition = [int(rng_rand.choice(f)) for f in feasible_doms]

        elif policy_name == "greedy":
            chosen_partition = plan.suggested_domains

        elif policy_name == "load_balance":
            lb_part = pick_load_balance_partition(plan, sub)
            if lb_part is None:
                structural += 1
                reject_reasons["lb_no_feasible"] += 1
                continue
            chosen_partition = lb_part

        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        # ── Commit chosen partition on REAL substrate ──
        plan_copy = PlanSummary(
            vnf_ids=plan.vnf_ids,
            required_tiers=plan.required_tiers,
            suggested_domains=chosen_partition,
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
            sub, sr, plan_copy, delays, mode="follow_prior",
        )

        if result.admitted and result.partition is not None:
            placement = build_placement_plan(sr, result)
            if placement is not None:
                sub.allocate(placement, sr)
                active_slices[sr.request_id] = (placement, sr)
                admitted += 1
            else:
                reject_reasons["incomplete_placement"] += 1
        else:
            # Classify rejection
            if result.retry_history.attempts:
                last = result.retry_history.attempts[-1]
                if last.violation:
                    v = last.violation
                    if v.actor_infeasible:
                        reject_reasons["actor_infeasible"] += 1
                    elif v.cross_domain_infeasible:
                        reject_reasons["cross_domain_bw"] += 1
                    elif v.c7_violated:
                        reject_reasons["c7_delay"] += 1
                    elif v.c9_violated:
                        reject_reasons["c9_hops"] += 1
                    else:
                        reject_reasons["other"] += 1
                else:
                    reject_reasons["unknown"] += 1
            else:
                reject_reasons["no_attempts"] += 1

    return total, admitted, structural, reject_reasons


def main():
    sub = build_substrate(SUBSTRATE_SEED)
    delays = build_delays(sub)

    logger.info("=" * 90)
    logger.info("DECISIVE TEST: Does myopic per-slice optimal survive dynamic accumulation?")
    logger.info("  Substrate: 5 domains x 3 nodes, 10%% edge/10%% MEC, seed=%d", SUBSTRATE_SEED)
    logger.info("  Arrivals: %d per seed, lambda=%.1f, mu=%.3f", NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE)
    logger.info("  Seeds: %s", ARRIVAL_SEEDS)
    logger.info("=" * 90)

    policies = ["oracle", "random", "greedy", "load_balance"]
    results = {p: [] for p in policies}
    reject_agg = {p: Counter() for p in policies}

    for seed in ARRIVAL_SEEDS:
        logger.info("")
        logger.info("--- Seed %d ---", seed)
        for policy in policies:
            t0 = time.time()
            total, admitted, structural, rejects = run_dynamic_policy(
                seed, sub, delays, policy
            )
            elapsed = time.time() - t0
            pct = 100 * admitted / total
            results[policy].append(pct)
            for k, v in rejects.items():
                reject_agg[policy][k] += v
            logger.info("  %-14s: %3d/%d (%5.1f%%)  structural=%d  time=%.1fs",
                        policy, admitted, total, pct, structural, elapsed)

    # ── Aggregate ──────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("AGGREGATE RESULTS (%d seeds)", len(ARRIVAL_SEEDS))
    logger.info("=" * 90)

    for policy in policies:
        pcts = results[policy]
        logger.info("  %-14s: %.1f%% +/- %.1f%%  [%s]",
                    policy,
                    np.mean(pcts), np.std(pcts),
                    ", ".join(f"{p:.1f}" for p in pcts))

    logger.info("")
    logger.info("REJECT REASON BREAKDOWN (total across all seeds):")
    for policy in policies:
        logger.info("  %s: %s", policy, dict(reject_agg[policy]))

    # ── Verdict ────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("VERDICT")
    logger.info("=" * 90)

    oracle_mean = np.mean(results["oracle"])
    random_mean = np.mean(results["random"])
    greedy_mean = np.mean(results["greedy"])
    lb_mean = np.mean(results["load_balance"])

    logger.info("  Random (strong baseline):  %.1f%%", random_mean)
    logger.info("  Greedy (myopic weak):      %.1f%%  (%.1f pp vs random)",
                greedy_mean, greedy_mean - random_mean)
    logger.info("  Load-balance:              %.1f%%  (%.1f pp vs random)",
                lb_mean, lb_mean - random_mean)
    logger.info("  Oracle (myopic optimal):   %.1f%%  (%.1f pp vs random)",
                oracle_mean, oracle_mean - random_mean)

    if oracle_mean > random_mean + 2.0:
        logger.info("")
        logger.info("  MYOPIC INTELLIGENCE TRANSFERS.")
        logger.info("  Per-slice partition quality survives dynamic accumulation.")
        logger.info("  The MDO's job is to approximate the per-slice oracle.")
        logger.info("  Bandit formulation may suffice if packing is not too aggressive.")
    elif oracle_mean > random_mean - 2.0:
        logger.info("")
        logger.info("  MYOPIC INTELLIGENCE IS MARGINAL.")
        logger.info("  Per-slice optimal is within noise of random dynamically.")
        logger.info("  The bandit formulation cannot beat random.")
        logger.info("  Sequential (MDP) reformulation is needed.")
    else:
        logger.info("")
        logger.info("  MYOPIC INTELLIGENCE SELF-DESTRUCTS.")
        logger.info("  Per-slice optimal loses to random dynamically (as greedy did).")
        logger.info("  Stronger myopic = worse dynamic. The job is sequential.")
        logger.info("  Bandit formulation is the bottleneck. MDP reformulation required.")

    # Load-balance assessment
    if lb_mean > random_mean + 2.0:
        logger.info("")
        logger.info("  LOAD-BALANCE beats random by %.1f pp.", lb_mean - random_mean)
        logger.info("  State-aware heuristic is a tractable dynamic ceiling candidate.")
    elif lb_mean > random_mean:
        logger.info("")
        logger.info("  LOAD-BALANCE is marginally better than random (%.1f pp).",
                    lb_mean - random_mean)
    else:
        logger.info("")
        logger.info("  LOAD-BALANCE does not beat random (%.1f pp).",
                    lb_mean - random_mean)

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
