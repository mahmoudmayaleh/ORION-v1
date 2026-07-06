#!/usr/bin/env python3
"""Dynamic ceiling: hindsight rollout + physical capacity analysis.

Tests whether sequential intelligence (future knowledge + rejection) can
beat random. If not, the dynamic ceiling is close to random and no policy
reformulation is worth building.

Policies tested:
  1. Random (strong baseline — currently best measured dynamic policy)
  2. Oracle (myopic optimal — per-slice best partition, no rejection)
  3. Hindsight-selective oracle: oracle + reject expensive arrivals (future knowledge)
  4. Hindsight-selective random: random + reject expensive arrivals (future knowledge)
  5. Utilization-gated oracle: oracle + reject if substrate util > threshold (no future knowledge)

The decomposition:
  - (3 vs 1): Total value of sequential + per-slice intelligence
  - (4 vs 1): Value of rejection alone (the sequential prize)
  - (3 vs 4): Value of per-slice intelligence ON TOP OF rejection
  - (5 vs 1): Value of state-aware admission gating (no future knowledge needed)

Also computes the physical capacity bound as a sanity check.
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
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
    sub_copy = copy.deepcopy(substrate)
    plan_copy = PlanSummary(
        vnf_ids=plan.vnf_ids, required_tiers=plan.required_tiers,
        suggested_domains=partition, cpu_demands=plan.cpu_demands,
        ram_demands=plan.ram_demands, vcrs=plan.vcrs, bw_demands=plan.bw_demands,
    )
    coord = MDOCoordinator(
        None, actors, MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
    )
    result = coord.resolve_arrival(sub_copy, slice_req, plan_copy, delays, mode="follow_prior")
    return result.admitted, result.total_cost, result


def build_placement_plan(slice_req, mdo_result):
    vnf_placements, cpu_alloc, ram_alloc = {}, {}, {}
    flow_routes, bw_alloc = {}, {}
    vnf_by_id = {v.vnf_id: v for v in slice_req.vnfs}
    for resp in mdo_result.domain_responses.values():
        for vnf_id, node_id in resp.placements.items():
            vnf_placements[vnf_id] = node_id
            cpu_alloc[vnf_id] = vnf_by_id[vnf_id].cpu_demand
            ram_alloc[vnf_id] = vnf_by_id[vnf_id].ram_demand
        for fk, links in resp.routes.items():
            flow_routes[fk] = links
            bw_alloc[fk] = {lid: resp.bw_allocated.get(fk, 0.0) for lid in links}
    for fk, links in mdo_result.cross_domain_routes.items():
        flow_routes[fk] = links
        bw = mdo_result.cross_domain_bw.get(fk, 0.0)
        bw_alloc[fk] = {lid: bw for lid in links}
    if any(v.vnf_id not in vnf_placements for v in slice_req.vnfs):
        return None
    return PlacementPlan(
        plan_id=f"{slice_req.request_id}_dyn",
        vnf_placements=vnf_placements, cpu_allocations=cpu_alloc,
        ram_allocations=ram_alloc, flow_routes=flow_routes,
        bw_allocations=bw_alloc, is_structurally_valid=True, source="dynamic",
    )


def substrate_utilization(sub):
    """Current CPU utilization across all nodes (0.0 to 1.0)."""
    total_cap = 0.0
    total_used = 0.0
    for _, d in sub.graph.nodes(data=True):
        if d.get("domain_id", -1) < 0:
            continue
        cap = float(d["cpu_capacity"])
        res = float(d["cpu_residual"])
        total_cap += cap
        total_used += (cap - res)
    return total_used / total_cap if total_cap > 0 else 0.0


@dataclass
class ArrivalInfo:
    """Pre-computed info about an arrival, used for hindsight scoring."""
    index: int
    request_id: str
    arrival_time: float
    departure_time: float
    lifetime: float
    total_cpu: float
    total_ram: float
    resource_time_cost: float  # cpu × lifetime
    num_vnfs: int


def extract_arrival_info(ap):
    """Extract arrival/departure info from event sequence for hindsight scoring."""
    arrivals = {}
    departures = {}
    for event in ap.events:
        if event.event_type == EventType.ARRIVAL and event.slice_request is not None:
            sr = event.slice_request
            cpu = sum(v.cpu_demand for v in sr.vnfs)
            ram = sum(v.ram_demand for v in sr.vnfs)
            arrivals[sr.request_id] = {
                "time": event.time, "cpu": cpu, "ram": ram,
                "num_vnfs": len(sr.vnfs), "request": sr,
            }
        elif event.event_type == EventType.DEPARTURE:
            departures[event.request_id] = event.time

    infos = []
    idx = 0
    for rid, a in arrivals.items():
        dep_time = departures.get(rid, a["time"] + 1000)
        lifetime = dep_time - a["time"]
        infos.append(ArrivalInfo(
            index=idx, request_id=rid, arrival_time=a["time"],
            departure_time=dep_time, lifetime=lifetime,
            total_cpu=a["cpu"], total_ram=a["ram"],
            resource_time_cost=a["cpu"] * lifetime,
            num_vnfs=a["num_vnfs"],
        ))
        idx += 1

    return infos


def find_oracle_partition(plan, slice_req, substrate, actors, delays, feasible_doms):
    """Find the best admitting partition via enumeration."""
    all_partitions = list(itertools.product(*feasible_doms))
    if len(all_partitions) > MAX_PARTITIONS_PER_SLICE:
        rng_sample = np.random.default_rng(hash(slice_req.request_id) % 2**32 + 999)
        indices = rng_sample.choice(len(all_partitions), MAX_PARTITIONS_PER_SLICE, replace=False)
        all_partitions = [all_partitions[i] for i in indices]

    best_part = None
    best_cost = float("inf")
    for part in all_partitions:
        part_list = list(part)
        adm, cost, _ = try_partition(part_list, plan, slice_req, substrate, actors, delays)
        if adm and cost < best_cost:
            best_part = part_list
            best_cost = cost
    return best_part


def run_dynamic(arrival_seed, sub_template, delays, policy_name,
                skip_set=None, util_threshold=1.0):
    """Run one dynamic episode.

    policy_name: "oracle" or "random"
    skip_set: set of request_ids to reject (hindsight knowledge)
    util_threshold: reject if substrate util > this (state-aware gating)
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
    total = admitted = structural = skipped = 0

    while ap.has_next():
        event = ap.next_event()
        if event.event_type == EventType.DEPARTURE:
            entry = active_slices.pop(event.request_id, None)
            if entry:
                sub.deallocate(entry[0], entry[1])
            continue

        total += 1
        sr = event.slice_request

        # Hindsight rejection
        if skip_set and sr.request_id in skip_set:
            skipped += 1
            continue

        # Utilization gating
        if substrate_utilization(sub) > util_threshold:
            skipped += 1
            continue

        plan = greedy_plan_builder(sr, sub)
        if plan is None:
            structural += 1
            continue

        feasible_doms = get_feasible_domains_per_vnf(plan, sub)
        if any(len(f) == 0 for f in feasible_doms):
            structural += 1
            continue

        if policy_name == "oracle":
            chosen = find_oracle_partition(plan, sr, sub, actors, delays, feasible_doms)
            if chosen is None:
                continue
        else:
            rng_rand = np.random.default_rng(hash(sr.request_id) % 2**32 + arrival_seed)
            chosen = [int(rng_rand.choice(f)) for f in feasible_doms]

        plan_copy = PlanSummary(
            vnf_ids=plan.vnf_ids, required_tiers=plan.required_tiers,
            suggested_domains=chosen, cpu_demands=plan.cpu_demands,
            ram_demands=plan.ram_demands, vcrs=plan.vcrs, bw_demands=plan.bw_demands,
        )
        coord = MDOCoordinator(
            None, actors, MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
        )
        result = coord.resolve_arrival(sub, sr, plan_copy, delays, mode="follow_prior")

        if result.admitted:
            placement = build_placement_plan(sr, result)
            if placement:
                sub.allocate(placement, sr)
                active_slices[sr.request_id] = (placement, sr)
                admitted += 1

    return total, admitted, structural, skipped


def compute_physical_bound(arrival_seed, sub_template):
    """Compute the physical capacity upper bound."""
    sub = copy.deepcopy(sub_template)

    # Total substrate resources (compute BEFORE passing to ArrivalProcess)
    total_cpu = 0.0
    total_ram = 0.0
    for nid, d in sub.graph.nodes(data=True):
        if "cpu_capacity" in d:
            total_cpu += float(d["cpu_capacity"])
            total_ram += float(d["ram_capacity"])

    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    arrival_infos = extract_arrival_info(ap)
    if not arrival_infos:
        return {"total_cpu": total_cpu, "total_ram": total_ram, "error": "no arrivals"}

    avg_cpu = np.mean([a.total_cpu for a in arrival_infos])
    avg_ram = np.mean([a.total_ram for a in arrival_infos])
    avg_lifetime = np.mean([a.lifetime for a in arrival_infos])
    avg_vnfs = np.mean([a.num_vnfs for a in arrival_infos])
    episode_dur = max(a.arrival_time for a in arrival_infos) - min(a.arrival_time for a in arrival_infos)

    max_sim_cpu = total_cpu / avg_cpu if avg_cpu > 0 else float("inf")
    max_sim_ram = total_ram / avg_ram if avg_ram > 0 else float("inf")
    offered_load = ARRIVAL_RATE * avg_lifetime
    theoretical_max = min(1.0, min(max_sim_cpu, max_sim_ram) / offered_load) if offered_load > 0 else 1.0

    return {
        "total_cpu": total_cpu, "total_ram": total_ram,
        "avg_cpu_per_slice": avg_cpu, "avg_ram_per_slice": avg_ram,
        "avg_lifetime": avg_lifetime, "avg_vnfs": avg_vnfs,
        "episode_duration": episode_dur,
        "max_simultaneous_cpu": max_sim_cpu, "max_simultaneous_ram": max_sim_ram,
        "offered_load": offered_load, "theoretical_max_admission": theoretical_max,
    }


def build_skip_sets(arrival_seed, sub_template, percentiles):
    """Build skip sets: reject the most expensive X% of arrivals."""
    sub = copy.deepcopy(sub_template)
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    infos = extract_arrival_info(ap)
    costs = sorted(infos, key=lambda a: a.resource_time_cost, reverse=True)

    skip_sets = {}
    for pct in percentiles:
        n_skip = int(len(costs) * pct / 100)
        skip_sets[pct] = {a.request_id for a in costs[:n_skip]}
    return skip_sets


def main():
    sub = build_substrate(SUBSTRATE_SEED)
    delays = build_delays(sub)

    logger.info("=" * 90)
    logger.info("DYNAMIC CEILING — hindsight rollout + physical capacity analysis")
    logger.info("  Substrate: 5 domains x 3 nodes, 10%% edge/10%% MEC")
    logger.info("  Arrivals: %d per seed, lambda=%.1f, mu=%.3f", NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE)
    logger.info("  Seeds: %s", ARRIVAL_SEEDS)
    logger.info("=" * 90)

    # ── PART 0: Physical Capacity Analysis ─────────────────────────────────

    logger.info("")
    logger.info("PART 0: PHYSICAL CAPACITY BOUND")
    logger.info("-" * 60)

    for seed in ARRIVAL_SEEDS[:1]:  # One seed suffices for averages
        stats = compute_physical_bound(seed, sub)
        logger.info("  Total substrate CPU: %.0f", stats["total_cpu"])
        logger.info("  Total substrate RAM: %.0f", stats["total_ram"])
        logger.info("  Avg CPU/slice: %.1f", stats["avg_cpu_per_slice"])
        logger.info("  Avg RAM/slice: %.1f", stats["avg_ram_per_slice"])
        logger.info("  Avg lifetime: %.1f time units", stats["avg_lifetime"])
        logger.info("  Avg VNFs/slice: %.1f", stats["avg_vnfs"])
        logger.info("  Episode duration: %.1f time units", stats["episode_duration"])
        max_sim = min(stats["max_simultaneous_cpu"], stats["max_simultaneous_ram"])
        logger.info("  Max simultaneous (CPU): %.1f slices", stats["max_simultaneous_cpu"])
        logger.info("  Max simultaneous (RAM): %.1f slices", stats["max_simultaneous_ram"])
        logger.info("  Offered load: %.1f simultaneous slices", stats["offered_load"])
        overload = stats["offered_load"] / max_sim if max_sim > 0 else float("inf")
        logger.info("  Overload factor: %.1fx", overload)
        logger.info("  Theoretical max admission: %.1f%%", 100 * stats["theoretical_max_admission"])

    # ── PART 1: Baseline policies ──────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("PART 1: BASELINE POLICIES (no future knowledge)")
    logger.info("=" * 90)

    random_results = []
    oracle_results = []

    for seed in ARRIVAL_SEEDS:
        t0 = time.time()
        _, adm_r, struct_r, _ = run_dynamic(seed, sub, delays, "random")
        random_results.append(100 * adm_r / NUM_ARRIVALS)
        _, adm_o, struct_o, _ = run_dynamic(seed, sub, delays, "oracle")
        oracle_results.append(100 * adm_o / NUM_ARRIVALS)
        logger.info("  Seed %d: random=%.1f%% (%d)  oracle=%.1f%% (%d)  [%.1fs]",
                    seed, random_results[-1], adm_r, oracle_results[-1], adm_o,
                    time.time() - t0)

    logger.info("  Random:  %.1f%% +/- %.1f%%", np.mean(random_results), np.std(random_results))
    logger.info("  Oracle:  %.1f%% +/- %.1f%%", np.mean(oracle_results), np.std(oracle_results))

    # ── PART 2: Hindsight-selective policies ────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("PART 2: HINDSIGHT-SELECTIVE (with future knowledge)")
    logger.info("  Reject the most expensive X%% of arrivals (by CPU × lifetime)")
    logger.info("=" * 90)

    rejection_pcts = [10, 20, 30, 40, 50, 60]

    for pct in rejection_pcts:
        hs_oracle = []
        hs_random = []
        for seed in ARRIVAL_SEEDS:
            skip_sets = build_skip_sets(seed, sub, [pct])
            skip = skip_sets[pct]
            _, adm_ho, _, sk_ho = run_dynamic(seed, sub, delays, "oracle", skip_set=skip)
            _, adm_hr, _, sk_hr = run_dynamic(seed, sub, delays, "random", skip_set=skip)
            hs_oracle.append(100 * adm_ho / NUM_ARRIVALS)
            hs_random.append(100 * adm_hr / NUM_ARRIVALS)
        logger.info("  Skip %2d%%: hindsight-oracle=%.1f%% +/- %.1f%%  "
                    "hindsight-random=%.1f%% +/- %.1f%%",
                    pct,
                    np.mean(hs_oracle), np.std(hs_oracle),
                    np.mean(hs_random), np.std(hs_random))

    # ── PART 3: Utilization-gated policies (no future knowledge) ───────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("PART 3: UTILIZATION-GATED (state-aware, no future knowledge)")
    logger.info("  Reject if substrate CPU utilization > threshold")
    logger.info("=" * 90)

    util_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for thresh in util_thresholds:
        ug_oracle = []
        ug_random = []
        for seed in ARRIVAL_SEEDS:
            _, adm_uo, _, sk_uo = run_dynamic(seed, sub, delays, "oracle", util_threshold=thresh)
            _, adm_ur, _, sk_ur = run_dynamic(seed, sub, delays, "random", util_threshold=thresh)
            ug_oracle.append(100 * adm_uo / NUM_ARRIVALS)
            ug_random.append(100 * adm_ur / NUM_ARRIVALS)
        logger.info("  Util <= %.0f%%: oracle=%.1f%% +/- %.1f%%  "
                    "random=%.1f%% +/- %.1f%%",
                    100 * thresh,
                    np.mean(ug_oracle), np.std(ug_oracle),
                    np.mean(ug_random), np.std(ug_random))

    # ── VERDICT ────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("VERDICT")
    logger.info("=" * 90)

    random_mean = np.mean(random_results)
    oracle_mean = np.mean(oracle_results)

    logger.info("  Random baseline:       %.1f%%", random_mean)
    logger.info("  Myopic oracle:         %.1f%%", oracle_mean)
    logger.info("  (Best hindsight and utilization-gated results above)")
    logger.info("")
    logger.info("  If ALL policies cluster within ~3 pp of random:")
    logger.info("    → The substrate is capacity-bound. No policy can do much better.")
    logger.info("    → The contribution is in the plan layer (reducing 26%% upstream kills).")
    logger.info("    → Or in a larger/different substrate where partition has room.")
    logger.info("")
    logger.info("  If hindsight-selective beats random by > 5 pp:")
    logger.info("    → Sequential intelligence has a real prize.")
    logger.info("    → MDP reformulation is worth building.")
    logger.info("    → The gap (hindsight - oracle) measures the value of rejection capability.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
