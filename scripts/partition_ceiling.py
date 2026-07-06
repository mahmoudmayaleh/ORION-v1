#!/usr/bin/env python3
"""Optimal-partition reference: brute-force enumerate feasible partitions.

For each arriving slice in the hard-corner regime:
  1. Build the plan (using greedy FFD as plan builder)
  2. Enumerate ALL feasible partitions (domain assignment per VNF)
  3. For each partition, run through actual coordinator dispatch on a substrate copy
  4. Record the best achievable outcome
  5. Compare random's partition quality to the per-slice optimum

Answers: does the partition decision have headroom, or is random near-optimal?
Also quantifies the plan-builder confound.
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
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
from orion.mdo.observation import build_mdo_observation, build_tier_masks
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_DOMAINS = 5
INTER_DOMAIN_BW = 200.0
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
NUM_ARRIVALS = 200


def build_substrate(seed=0):
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
    """For each VNF, find which domains have at least one node of the
    required tier with enough CPU/RAM residual."""
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
    """Try a specific partition using the actual coordinator dispatch on a deep copy.

    Returns (admitted, cost, e2e_delay, violations) or (False, inf, inf, 0).
    The substrate is deep-copied so the original is never mutated.
    """
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
class SliceResult:
    slice_id: str = ""
    num_vnfs: int = 0
    plan_failed: bool = False
    plan_fail_reason: str = ""
    num_feasible_partitions: int = 0
    num_total_partitions: int = 0
    optimal_admitted: bool = False
    optimal_cost: float = float("inf")
    optimal_delay: float = float("inf")
    optimal_violations: int = 0
    optimal_partition: list[int] = field(default_factory=list)
    random_admitted: bool = False
    random_cost: float = float("inf")
    random_delay: float = float("inf")
    random_violations: int = 0
    random_partition: list[int] = field(default_factory=list)
    forced_split: bool = False
    n_unique_feasible_doms: int = 0


def main():
    sub = build_substrate(0)
    delays = build_delays(sub)
    g = sub.graph

    logger.info("=" * 90)
    logger.info("OPTIMAL PARTITION REFERENCE — brute-force enumeration")
    logger.info("  Hard-corner regime: 5 domains × 3 nodes, 10%% edge/10%% MEC")
    logger.info("  λ=%.1f, %d arrivals", ARRIVAL_RATE, NUM_ARRIVALS)
    logger.info("=" * 90)

    for dom in range(NUM_DOMAINS):
        tiers = {}
        cpu_total = 0
        for nid, d in g.nodes(data=True):
            if d["domain_id"] == dom:
                t = d["tier"]
                tiers[t] = tiers.get(t, 0) + 1
                cpu_total += float(d["cpu_capacity"])
        tier_str = ", ".join(f"{t}:{c}" for t, c in sorted(tiers.items()))
        logger.info("  Domain %d: %s  (%.0f CPU total)", dom, tier_str, cpu_total)

    rng = np.random.default_rng(42)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    actors = {}
    for d in range(NUM_DOMAINS):
        torch.manual_seed(d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))

    results: list[SliceResult] = []

    # Use a FIXED substrate snapshot for all partition evaluations
    # (no dynamic resource consumption — pure per-slice comparison)
    sub_snapshot = copy.deepcopy(sub)

    arrival_count = 0
    for event in ap.events:
        if event.event_type == "departure" or event.slice_request is None:
            continue

        arrival_count += 1
        sr = SliceResult(
            slice_id=event.slice_request.request_id,
            num_vnfs=len(event.slice_request.vnfs),
        )

        plan = greedy_plan_builder(event.slice_request, sub_snapshot)
        if plan is None:
            sr.plan_failed = True
            result_ffd = _run_greedy_ffd(sub_snapshot, event.slice_request, GreedyConfig())
            sr.plan_fail_reason = result_ffd.fail_reason
            results.append(sr)
            continue

        feasible_doms = get_feasible_domains_per_vnf(plan, sub_snapshot)

        if any(len(f) == 0 for f in feasible_doms):
            sr.plan_failed = True
            sr.plan_fail_reason = "no feasible domain for some VNF"
            results.append(sr)
            continue

        # Check if this slice has a forced split
        all_same = all(set(f) == set(feasible_doms[0]) for f in feasible_doms)
        can_be_single = any(
            all(d in f for f in feasible_doms)
            for d in range(NUM_DOMAINS)
        )
        sr.forced_split = not can_be_single
        sr.n_unique_feasible_doms = len(set().union(*feasible_doms))

        all_partitions = list(itertools.product(*feasible_doms))
        sr.num_total_partitions = len(all_partitions)

        if len(all_partitions) > 2000:
            logger.warning("  %s: %d partitions — too many, sampling 2000",
                           sr.slice_id, len(all_partitions))
            rng_sample = np.random.default_rng(hash(sr.slice_id) % 2**32 + 999)
            indices = rng_sample.choice(len(all_partitions), size=2000, replace=False)
            all_partitions = [all_partitions[i] for i in indices]

        best_admitted = False
        best_cost = float("inf")
        best_delay = float("inf")
        best_viols = 999
        best_part = []
        n_feasible = 0

        for part in all_partitions:
            part_list = list(part)
            admitted, cost, delay, viols = try_partition_via_coordinator(
                part_list, plan, event.slice_request, sub_snapshot, actors, delays,
            )
            if admitted:
                n_feasible += 1
                better = False
                if not best_admitted:
                    better = True
                elif viols < best_viols:
                    better = True
                elif viols == best_viols and cost < best_cost:
                    better = True
                if better:
                    best_admitted = True
                    best_cost = cost
                    best_delay = delay
                    best_viols = viols
                    best_part = part_list

        sr.num_feasible_partitions = n_feasible
        sr.optimal_admitted = best_admitted
        sr.optimal_cost = best_cost
        sr.optimal_delay = best_delay
        sr.optimal_violations = best_viols
        sr.optimal_partition = best_part

        # Random partition
        rng_rand = np.random.default_rng(hash(sr.slice_id) % 2**32)
        rand_part = [int(rng_rand.choice(f)) for f in feasible_doms]
        admitted_r, cost_r, delay_r, viols_r = try_partition_via_coordinator(
            rand_part, plan, event.slice_request, sub_snapshot, actors, delays,
        )
        sr.random_admitted = admitted_r
        sr.random_cost = cost_r
        sr.random_delay = delay_r
        sr.random_violations = viols_r
        sr.random_partition = rand_part

        results.append(sr)

        if arrival_count % 20 == 0:
            logger.info("  processed %d/%d arrivals (%d plan-failed so far)",
                        arrival_count, NUM_ARRIVALS, sum(1 for r in results if r.plan_failed))

    # ── Analysis ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info("RESULTS")
    logger.info("=" * 90)

    total = len(results)
    plan_failed = sum(1 for r in results if r.plan_failed)
    plan_survived = total - plan_failed

    logger.info("")
    logger.info("PLAN-BUILDER CONFOUND:")
    logger.info("  Total arrivals:    %d", total)
    logger.info("  Plan failed:       %d (%.1f%%)", plan_failed, 100 * plan_failed / max(total, 1))
    logger.info("  Plan survived:     %d (%.1f%%)", plan_survived, 100 * plan_survived / max(total, 1))

    fail_reasons = {}
    for r in results:
        if r.plan_failed:
            key = r.plan_fail_reason[:50]
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        logger.info("    %s: %d", reason, count)

    survived = [r for r in results if not r.plan_failed]
    if not survived:
        logger.info("  No slices survived plan construction.")
        return

    logger.info("")
    logger.info("PARTITION ENUMERATION (on %d surviving slices):", len(survived))

    opt_admitted = [r for r in survived if r.optimal_admitted]
    rand_admitted = [r for r in survived if r.random_admitted]
    both_admitted = [r for r in survived if r.optimal_admitted and r.random_admitted]
    opt_only = [r for r in survived if r.optimal_admitted and not r.random_admitted]
    rand_only = [r for r in survived if r.random_admitted and not r.optimal_admitted]
    forced_split = [r for r in survived if r.forced_split]

    logger.info("  Forced cross-domain split:         %d/%d (%.1f%%)",
                len(forced_split), len(survived), 100 * len(forced_split) / len(survived))

    avg_partitions = np.mean([r.num_total_partitions for r in survived])
    avg_feasible = np.mean([r.num_feasible_partitions for r in survived])
    logger.info("  Avg total partitions per slice:     %.1f", avg_partitions)
    logger.info("  Avg feasible partitions per slice:  %.1f", avg_feasible)
    logger.info("")
    logger.info("  Optimal admits:  %d/%d (%.1f%%)",
                len(opt_admitted), len(survived), 100 * len(opt_admitted) / len(survived))
    logger.info("  Random admits:   %d/%d (%.1f%%)",
                len(rand_admitted), len(survived), 100 * len(rand_admitted) / len(survived))
    logger.info("  Optimal-only:    %d (partitions where optimal admits but random fails)",
                len(opt_only))
    logger.info("  Random-only:     %d (should be 0 if enumeration is exhaustive)",
                len(rand_only))

    # Partition enumeration stats
    partition_counts = [r.num_total_partitions for r in survived]
    feasible_counts = [r.num_feasible_partitions for r in survived]
    logger.info("")
    logger.info("  Partition counts: min=%d  max=%d  median=%.0f",
                min(partition_counts), max(partition_counts), np.median(partition_counts))
    logger.info("  Feasible counts:  min=%d  max=%d  median=%.0f",
                min(feasible_counts), max(feasible_counts), np.median(feasible_counts))

    # ── Cost/delay comparison on slices where both admit ────────────────────
    logger.info("")
    logger.info("PARTITION QUALITY (on %d slices where both optimal AND random admit):", len(both_admitted))

    if both_admitted:
        cost_gaps = [(r.random_cost - r.optimal_cost) for r in both_admitted]
        delay_gaps = [(r.random_delay - r.optimal_delay) for r in both_admitted]
        viol_gaps = [(r.random_violations - r.optimal_violations) for r in both_admitted]

        cost_ratios = [r.random_cost / max(r.optimal_cost, 0.01) for r in both_admitted]

        logger.info("  Cost:  random=%.1f  optimal=%.1f  Δ=%.1f  ratio=%.3f",
                    np.mean([r.random_cost for r in both_admitted]),
                    np.mean([r.optimal_cost for r in both_admitted]),
                    np.mean(cost_gaps),
                    np.mean(cost_ratios))
        logger.info("  Delay: random=%.1f  optimal=%.1f  Δ=%.1f",
                    np.mean([r.random_delay for r in both_admitted]),
                    np.mean([r.optimal_delay for r in both_admitted]),
                    np.mean(delay_gaps))
        logger.info("  Viols: random=%.2f  optimal=%.2f  Δ=%.2f",
                    np.mean([r.random_violations for r in both_admitted]),
                    np.mean([r.optimal_violations for r in both_admitted]),
                    np.mean(viol_gaps))

        pct_cost_same = sum(1 for g in cost_gaps if abs(g) < 0.01) / len(cost_gaps)
        pct_within_5 = sum(1 for r in cost_ratios if r < 1.05) / len(cost_ratios)
        pct_within_10 = sum(1 for r in cost_ratios if r < 1.10) / len(cost_ratios)
        logger.info("  %% random = optimal (cost): %.1f%%", 100 * pct_cost_same)
        logger.info("  %% random within 5%% of optimal (cost): %.1f%%", 100 * pct_within_5)
        logger.info("  %% random within 10%% of optimal (cost): %.1f%%", 100 * pct_within_10)
    else:
        logger.info("  (no slices where both admit)")

    # ── Admission headroom ──────────────────────────────────────────────────
    logger.info("")
    logger.info("ADMISSION HEADROOM:")
    logger.info("  Plan builder kills:         %d/%d = %.1f%%",
                plan_failed, total, 100 * plan_failed / total)
    opt_admit_pct = 100 * len(opt_admitted) / len(survived) if survived else 0
    rand_admit_pct = 100 * len(rand_admitted) / len(survived) if survived else 0
    logger.info("  Of survivors, optimal:      %d/%d = %.1f%%",
                len(opt_admitted), len(survived), opt_admit_pct)
    logger.info("  Of survivors, random:       %d/%d = %.1f%%",
                len(rand_admitted), len(survived), rand_admit_pct)
    logger.info("  Partition admission gap:    %.1f pp (%d slices)",
                opt_admit_pct - rand_admit_pct, len(opt_only))
    logger.info("  Overall optimal admission:  %.1f%%", 100 * len(opt_admitted) / total)
    logger.info("  Overall random admission:   %.1f%%", 100 * len(rand_admitted) / total)

    # ── Verdict ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 90)
    logger.info("VERDICT:")
    headroom_pp = opt_admit_pct - rand_admit_pct

    if len(opt_only) == 0:
        if both_admitted and np.mean(cost_ratios) < 1.05:
            logger.info("  FORK B CONFIRMED: random is near-optimal per slice.")
            logger.info("  Partition has no admission headroom and <=5%% cost gap.")
            logger.info("  Value must come from upstream: better plans, not better partitions.")
        elif both_admitted:
            logger.info("  Partition has COST headroom (ratio=%.3f) but NO admission headroom.",
                        np.mean(cost_ratios))
            logger.info("  A smarter partition saves cost but does not admit more slices.")
        else:
            logger.info("  Insufficient data to judge (no both-admitted slices).")
    elif headroom_pp > 5:
        logger.info("  PARTITION HAS HEADROOM: %.1f pp gap, %d slices where optimal admits",
                    headroom_pp, len(opt_only))
        logger.info("  but random fails. The learned policy is broken, not the problem.")
    elif headroom_pp > 0:
        logger.info("  SMALL HEADROOM: %.1f pp gap (%d slices). Partition matters modestly.",
                    headroom_pp, len(opt_only))
        logger.info("  But the plan-builder confound (%.1f%% killed upstream) is the bigger lever.",
                    100 * plan_failed / total)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
