#!/usr/bin/env python3
"""Item 2: Per-instance ceilings and ColocFB residual composition.

For each generated topology instance across all 8 families:
  1. Run ColocFB on fresh substrate, count kills
  2. Run the frozen kill classifier (full verifier gate) on each ColocFB kill
  3. Report per-family: ceiling, ColocFB admission, residual composition

Output: the residual-composition table that sizes the memory prize.
The residual = ceiling - ColocFB. Its composition (search failure vs
C5b trap vs C7 trap vs physical) tells us what M^B is chasing.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from orion.actors.routing import route_cross_domain_flow, allocate_route_bw, deallocate_route_bw
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.delay_model import node_sojourn, link_sojourn
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_families import (
    ALL_FAMILIES, TRAIN_FAMILIES, TEST_FAMILIES,
    TopologyFamily, generate_family_instance, compute_signature,
)
from orion.types import InfrastructureTier

from plan_builder_bracket import _run_colocation_ffd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_ARRIVALS = 200
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
INSTANCE_SEEDS = [0, 1, 2]
ARRIVAL_SEED = 42
MAX_NODE_COMBOS = 5000


# ── Kill classifier (same gate as verifier, from kill_classification.py v2) ──


def get_feasible_nodes_per_vnf(slice_req, substrate):
    g = substrate.graph
    feasible = []
    for vnf in slice_req.vnfs:
        permitted_tiers = set()
        for nid in vnf.permitted_nodes:
            if nid in g.nodes:
                permitted_tiers.add(g.nodes[nid]["tier"])
        nodes = []
        for nid, d in g.nodes(data=True):
            if d.get("domain_id", -1) < 0:
                continue
            if d["tier"] not in permitted_tiers:
                continue
            if float(d["cpu_residual"]) >= vnf.cpu_demand and \
               float(d["ram_residual"]) >= vnf.ram_demand:
                nodes.append(nid)
        feasible.append(nodes)
    return feasible


def compute_e2e_delay(placement, routes, slice_req, substrate):
    g = substrate.graph
    vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(slice_req.vnfs)}
    extra_cpu = {}
    for vnf in slice_req.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        extra_cpu[nid] = extra_cpu.get(nid, 0.0) + vnf.cpu_demand

    total_delay = 0.0
    for vnf in slice_req.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        node = g.nodes[nid]
        cpu_cap = float(node["cpu_capacity"])
        cpu_used = cpu_cap - float(node["cpu_residual"]) + extra_cpu[nid]
        sojourn = node_sojourn(float(node["processing_delay"]), vnf.computational_intensity, cpu_cap, cpu_used)
        if math.isinf(sojourn):
            return math.inf
        total_delay += sojourn

    extra_bw = {}
    for fe in slice_req.flow_edges:
        fk = (fe.source_vnf, fe.target_vnf)
        for lid in routes.get(fk, []):
            extra_bw[lid] = extra_bw.get(lid, 0.0) + fe.bandwidth_demand

    for fe in slice_req.flow_edges:
        fk = (fe.source_vnf, fe.target_vnf)
        for lid in routes.get(fk, []):
            for u, v, d in g.edges(data=True):
                if d["link_id"] == lid:
                    bw_cap = float(d.get("bandwidth_capacity", d.get("bw_capacity", 1000)))
                    bw_used = bw_cap - float(d["bw_residual"]) + extra_bw.get(lid, 0.0)
                    sojourn = link_sojourn(float(d["propagation_delay"]), bw_cap, bw_used)
                    if math.isinf(sojourn):
                        return math.inf
                    total_delay += sojourn
                    break
    return total_delay


def try_placement_full(placement, slice_req, substrate, max_hops=3):
    g = substrate.graph
    node_cpu, node_ram = {}, {}
    for j, vnf in enumerate(slice_req.vnfs):
        nid = placement[j]
        node_cpu[nid] = node_cpu.get(nid, 0.0) + vnf.cpu_demand
        node_ram[nid] = node_ram.get(nid, 0.0) + vnf.ram_demand
        if node_cpu[nid] > float(g.nodes[nid]["cpu_residual"]) + 0.01:
            return False, "capacity"
        if node_ram[nid] > float(g.nodes[nid]["ram_residual"]) + 0.01:
            return False, "capacity"

    vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(slice_req.vnfs)}
    sub_copy = copy.deepcopy(substrate)
    routes = {}

    for fe in slice_req.flow_edges:
        src_node = vnf_to_node[fe.source_vnf]
        dst_node = vnf_to_node[fe.target_vnf]
        if src_node == dst_node:
            continue
        src_dom = g.nodes[src_node]["domain_id"]
        dst_dom = g.nodes[dst_node]["domain_id"]

        if src_dom == dst_dom:
            domain_nodes = [n for n, d in g.nodes(data=True) if d.get("domain_id") == src_dom]
            subg = sub_copy.graph.subgraph(domain_nodes)
            path = None
            try:
                for p in nx.shortest_simple_paths(subg, src_node, dst_node, weight="propagation_delay"):
                    ok = all(float(sub_copy.graph[p[i]][p[i+1]]["bw_residual"]) >= fe.bandwidth_demand
                             for i in range(len(p) - 1))
                    if ok:
                        path = p
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            if path is None:
                return False, "c5b_routing"
            link_ids = []
            for i in range(len(path) - 1):
                sub_copy.graph[path[i]][path[i+1]]["bw_residual"] -= fe.bandwidth_demand
                link_ids.append(sub_copy.graph[path[i]][path[i+1]]["link_id"])
            routes[(fe.source_vnf, fe.target_vnf)] = link_ids
        else:
            result = route_cross_domain_flow(sub_copy, src_node, dst_node,
                                              bw_demand=fe.bandwidth_demand, delay_budget=999999.0)
            if not result.feasible:
                return False, "c5b_routing"
            allocate_route_bw(sub_copy, result.path_links, fe.bandwidth_demand)
            routes[(fe.source_vnf, fe.target_vnf)] = result.path_links

    # C9
    hops = 0
    for fk, lids in routes.items():
        for lid in lids:
            for u, v, d in g.edges(data=True):
                if d["link_id"] == lid:
                    if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
                        hops += 1
                    break
    if hops > max_hops:
        return False, "c9_hops"

    # C7
    e2e = compute_e2e_delay(placement, routes, slice_req, substrate)
    if e2e > slice_req.qos.max_e2e_delay:
        return False, "c7_delay"

    return True, None


def classify_kill(slice_req, substrate):
    feasible_nodes = get_feasible_nodes_per_vnf(slice_req, substrate)
    for i, nodes in enumerate(feasible_nodes):
        if len(nodes) == 0:
            return "physical"

    combos = list(itertools.product(*feasible_nodes))
    if len(combos) > MAX_NODE_COMBOS:
        rng = np.random.default_rng(hash(slice_req.request_id) % 2**32)
        indices = rng.choice(len(combos), MAX_NODE_COMBOS, replace=False)
        combos = [combos[i] for i in indices]

    any_placement = False
    fail_reasons = Counter()

    for combo in combos:
        feasible, reason = try_placement_full(list(combo), slice_req, substrate)
        if feasible:
            return "search_failure"
        if reason == "capacity":
            continue
        any_placement = True
        fail_reasons[reason] += 1

    if not any_placement:
        return "physical"

    if fail_reasons.get("c5b_routing", 0) > 0 and fail_reasons.get("c7_delay", 0) == 0:
        return "c5b_trap"
    if fail_reasons.get("c7_delay", 0) > 0 and fail_reasons.get("c5b_routing", 0) == 0:
        return "c7_trap"
    return "mixed_trap"


# ── Main ───────────────────────────────────────────────────────────────────


def run_instance(substrate, arrival_seed):
    """Run ColocFB + classify kills on one instance."""
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    cfg = GreedyConfig()
    total = 0
    coloc_admitted = 0
    coloc_kills = Counter()
    ceiling_admitted = 0  # enumerator ceiling (any valid placement exists)

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        total += 1
        sr = event.slice_request

        # ColocFB
        result = _run_colocation_ffd(substrate, sr, cfg)
        if result.feasible:
            coloc_admitted += 1
            ceiling_admitted += 1  # if ColocFB admits, ceiling admits too
        else:
            # Classify the kill
            bin_name = classify_kill(sr, substrate)
            coloc_kills[bin_name] += 1
            if bin_name == "search_failure":
                ceiling_admitted += 1  # enumerator found a valid placement

    return total, coloc_admitted, ceiling_admitted, coloc_kills


def main():
    logger.info("=" * 90)
    logger.info("ITEM 2: Per-instance ceilings and ColocFB residual composition")
    logger.info("  %d families, %d instances each, %d arrivals, arrival_seed=%d",
                len(ALL_FAMILIES), len(INSTANCE_SEEDS), NUM_ARRIVALS, ARRIVAL_SEED)
    logger.info("=" * 90)

    family_results = {}

    for family in ALL_FAMILIES:
        t0 = time.time()
        totals = {"total": 0, "coloc_adm": 0, "ceiling_adm": 0}
        kill_bins = Counter()

        for inst_seed in INSTANCE_SEEDS:
            sub = generate_family_instance(family, seed=inst_seed)
            total, coloc_adm, ceil_adm, kills = run_instance(sub, ARRIVAL_SEED)
            totals["total"] += total
            totals["coloc_adm"] += coloc_adm
            totals["ceiling_adm"] += ceil_adm
            for k, v in kills.items():
                kill_bins[k] += v

        elapsed = time.time() - t0
        t = totals["total"]
        coloc_pct = 100 * totals["coloc_adm"] / t if t > 0 else 0
        ceil_pct = 100 * totals["ceiling_adm"] / t if t > 0 else 0
        residual = ceil_pct - coloc_pct
        coloc_kills_total = sum(kill_bins.values())

        family_results[family.short_name] = {
            "coloc_pct": coloc_pct,
            "ceiling_pct": ceil_pct,
            "residual_pp": residual,
            "kills": coloc_kills_total,
            "bins": dict(kill_bins),
        }

        logger.info("")
        logger.info("%-12s  ceiling=%.1f%%  ColocFB=%.1f%%  residual=%.1f pp  (%.1fs)",
                    family.short_name, ceil_pct, coloc_pct, residual, elapsed)
        if coloc_kills_total > 0:
            for bin_name in ["physical", "search_failure", "c5b_trap", "c7_trap", "mixed_trap"]:
                count = kill_bins.get(bin_name, 0)
                if count > 0:
                    pct_of_kills = 100 * count / coloc_kills_total
                    logger.info("    %-16s: %3d  (%.1f%% of %d kills)",
                                bin_name, count, pct_of_kills, coloc_kills_total)

    # ── Summary table ──────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("RESIDUAL COMPOSITION TABLE")
    logger.info("=" * 90)
    logger.info("  %-12s  %7s  %7s  %8s  %8s  %8s  %8s  %8s",
                "Family", "Ceiling", "ColocFB", "Residual", "Physical", "Search", "C5b", "C7/Mix")
    logger.info("  " + "-" * 85)

    total_residual = 0
    total_search = 0
    total_physical = 0
    total_trap = 0

    for family in ALL_FAMILIES:
        r = family_results[family.short_name]
        kills = r["kills"]
        phys = r["bins"].get("physical", 0)
        search = r["bins"].get("search_failure", 0)
        c5b = r["bins"].get("c5b_trap", 0)
        c7mix = r["bins"].get("c7_trap", 0) + r["bins"].get("mixed_trap", 0)

        total_residual += r["residual_pp"]
        total_search += search
        total_physical += phys
        total_trap += c5b + c7mix

        logger.info("  %-12s  %6.1f%%  %6.1f%%  %+7.1f pp  %7d  %7d  %7d  %7d",
                    family.short_name,
                    r["ceiling_pct"], r["coloc_pct"], r["residual_pp"],
                    phys, search, c5b, c7mix)

    logger.info("")
    logger.info("  Total residual across families: %.1f pp (avg %.1f pp/family)",
                total_residual, total_residual / len(ALL_FAMILIES))
    logger.info("  Residual composition: physical=%d, search_failure=%d, traps=%d",
                total_physical, total_search, total_trap)

    # ── Verdict ────────────────────────────────────────────────────────────

    logger.info("")
    logger.info("=" * 90)
    logger.info("MEMORY PRIZE ASSESSMENT")
    logger.info("=" * 90)

    if total_search > 0:
        logger.info("  Search failures in ColocFB residual: %d", total_search)
        logger.info("  These are placements ColocFB misses that a smarter builder finds.")
        logger.info("  M^B's job: learn which plan shapes (strategy, tier assignment,")
        logger.info("  cut points) survive per topology signature.")
    if total_trap > 0:
        logger.info("  C5b/C7 traps: %d", total_trap)
        logger.info("  No ordering finds these — require topology+load awareness.")
        logger.info("  This is the memory-shaped residual.")
    if total_physical > 0:
        logger.info("  Physical kills: %d (no policy can recover these)", total_physical)

    b_minus_families = [f for f in ALL_FAMILIES if "B-" in f.short_name]
    b_minus_residual = sum(family_results[f.short_name]["residual_pp"] for f in b_minus_families)
    logger.info("")
    logger.info("  B- family residual (where memory prize lives): %.1f pp total",
                b_minus_residual)
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
