#!/usr/bin/env python3
"""Kill classification: four bins for every structural kill.

For each arrival the greedy FFD plan builder kills on a FRESH substrate,
classify into exactly one of four bins:

  1. PHYSICAL:       No tier-feasible node assignment exists anywhere.
  2. SEARCH_FAILURE: A feasible assignment exists that passes the FULL
                     verifier (C2, C3, C5, C5b VCR, C7 M/M/1, C9), but
                     greedy FFD missed it.
  3. C5B_TRAP:       Node placements exist but all fail BW routing.
  4. C7_TRAP:        Node placements exist and route, but all fail the
                     load-dependent M/M/1 delay check (C7).

The enumerator applies the SAME gate as the verifier: load-dependent delay
is evaluated against the actual substrate state at arrival time (on a fresh
substrate for the static analysis). This is the arrival-state ceiling, not
the empty-substrate ceiling.

DESIGN DECISION (documented): delay is evaluated against the substrate
state at the time of arrival, including any allocations from prior slices
in the same episode. For the fresh-substrate static analysis this equals
the empty-substrate state. For the dynamic per-instance ceiling (used
later as the headline normalizer), this is the arrival-state.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.routing import route_cross_domain_flow, allocate_route_bw, deallocate_route_bw
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.delay_model import node_sojourn, link_sojourn
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
SUBSTRATE_SEED = 0
ARRIVAL_SEEDS = [42, 123, 456, 789, 1001]
MAX_NODE_COMBOS = 5000


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


def get_feasible_nodes_per_vnf(slice_req, substrate):
    """For each VNF, list all nodes with matching tier AND enough CPU/RAM."""
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
    """Compute load-dependent M/M/1 E2E delay for a placement.

    Same formula as verifier._compute_ground_truth_e2e, applied to the
    substrate state AFTER this slice's resources would be allocated.

    Args:
        placement: list of node IDs, one per VNF
        routes: dict of (src_vnf, dst_vnf) -> list of link_ids
        slice_req: the slice request
        substrate: substrate in arrival-time state (pre-allocation)

    Returns:
        E2E delay in ms. +inf if any resource is saturated.
    """
    g = substrate.graph

    # Simulate the allocation's effect on node loads
    vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(slice_req.vnfs)}

    # Compute per-node CPU that would be used AFTER allocating this slice
    extra_cpu = {}
    for vnf in slice_req.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        extra_cpu[nid] = extra_cpu.get(nid, 0.0) + vnf.cpu_demand

    total_delay = 0.0

    # Node sojourns
    for vnf in slice_req.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        node = g.nodes[nid]
        cpu_cap = float(node["cpu_capacity"])
        cpu_used_before = cpu_cap - float(node["cpu_residual"])
        cpu_used_after = cpu_used_before + extra_cpu[nid]

        sojourn = node_sojourn(
            base_processing_delay=float(node["processing_delay"]),
            intensity=vnf.computational_intensity,
            cpu_capacity=cpu_cap,
            cpu_used=cpu_used_after,
        )
        if math.isinf(sojourn):
            return math.inf
        total_delay += sojourn

    # Link sojourns for all routed flows
    extra_bw = {}
    for fe in slice_req.flow_edges:
        fk = (fe.source_vnf, fe.target_vnf)
        link_ids = routes.get(fk, [])
        for lid in link_ids:
            extra_bw[lid] = extra_bw.get(lid, 0.0) + fe.bandwidth_demand

    for fe in slice_req.flow_edges:
        fk = (fe.source_vnf, fe.target_vnf)
        link_ids = routes.get(fk, [])
        for lid in link_ids:
            # Find the edge for this link_id
            for u, v, d in g.edges(data=True):
                if d["link_id"] == lid:
                    bw_cap = float(d.get("bandwidth_capacity", d.get("bw_capacity", 1000)))
                    bw_used_before = bw_cap - float(d["bw_residual"])
                    bw_used_after = bw_used_before + extra_bw.get(lid, 0.0)

                    sojourn = link_sojourn(
                        propagation_delay=float(d["propagation_delay"]),
                        bandwidth_capacity=bw_cap,
                        bandwidth_used=bw_used_after,
                    )
                    if math.isinf(sojourn):
                        return math.inf
                    total_delay += sojourn
                    break

    return total_delay


def count_inter_domain_hops(routes, substrate):
    """Count inter-domain link traversals across all flow routes."""
    g = substrate.graph
    hops = 0
    for fk, link_ids in routes.items():
        for lid in link_ids:
            for u, v, d in g.edges(data=True):
                if d["link_id"] == lid:
                    u_dom = g.nodes[u].get("domain_id", -1)
                    v_dom = g.nodes[v].get("domain_id", -1)
                    if u_dom != v_dom:
                        hops += 1
                    break
    return hops


def try_placement_full_check(placement, slice_req, substrate, max_hops=3):
    """Try a specific node assignment with FULL verifier-equivalent checks.

    Returns: (feasible, fail_reason)
    - feasible: True if placement + routing passes C2/C3/C5/C5b/C7/C9
    - fail_reason: None if feasible, else "c5b_routing" or "c7_delay" or "c9_hops" or "capacity"
    """
    import networkx as nx
    g = substrate.graph

    # Check node capacity (multiple VNFs on same node)
    node_cpu = {}
    node_ram = {}
    for j, vnf in enumerate(slice_req.vnfs):
        nid = placement[j]
        node_cpu[nid] = node_cpu.get(nid, 0.0) + vnf.cpu_demand
        node_ram[nid] = node_ram.get(nid, 0.0) + vnf.ram_demand
        if node_cpu[nid] > float(g.nodes[nid]["cpu_residual"]) + 0.01:
            return False, "capacity"
        if node_ram[nid] > float(g.nodes[nid]["ram_residual"]) + 0.01:
            return False, "capacity"

    vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(slice_req.vnfs)}

    # Route all flows (intra + cross domain)
    sub_copy = copy.deepcopy(substrate)
    routes = {}
    all_ok = True
    fail_reason = None

    for fe in slice_req.flow_edges:
        src_node = vnf_to_node[fe.source_vnf]
        dst_node = vnf_to_node[fe.target_vnf]

        if src_node == dst_node:
            continue

        src_dom = g.nodes[src_node]["domain_id"]
        dst_dom = g.nodes[dst_node]["domain_id"]

        if src_dom == dst_dom:
            # Intra-domain: route on domain subgraph
            domain_nodes = [n for n, d in g.nodes(data=True)
                           if d.get("domain_id") == src_dom]
            subg = sub_copy.graph.subgraph(domain_nodes)
            path = None
            try:
                for p in nx.shortest_simple_paths(subg, src_node, dst_node,
                                                   weight="propagation_delay"):
                    ok = True
                    for i in range(len(p) - 1):
                        edge = sub_copy.graph[p[i]][p[i+1]]
                        if float(edge["bw_residual"]) < fe.bandwidth_demand:
                            ok = False
                            break
                    if ok:
                        path = p
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

            if path is None:
                all_ok = False
                fail_reason = "c5b_routing"
                break

            link_ids = []
            for i in range(len(path) - 1):
                sub_copy.graph[path[i]][path[i+1]]["bw_residual"] -= fe.bandwidth_demand
                link_ids.append(sub_copy.graph[path[i]][path[i+1]]["link_id"])
            routes[(fe.source_vnf, fe.target_vnf)] = link_ids
        else:
            # Cross-domain: route on full graph
            result = route_cross_domain_flow(
                sub_copy, src_node, dst_node,
                bw_demand=fe.bandwidth_demand,
                delay_budget=999999.0,  # Don't reject on delay here; check below
            )
            if not result.feasible:
                all_ok = False
                fail_reason = "c5b_routing"
                break
            allocate_route_bw(sub_copy, result.path_links, fe.bandwidth_demand)
            routes[(fe.source_vnf, fe.target_vnf)] = result.path_links

    if not all_ok:
        return False, fail_reason

    # C9: inter-domain hops
    hops = count_inter_domain_hops(routes, substrate)
    if hops > max_hops:
        return False, "c9_hops"

    # C7: load-dependent M/M/1 delay
    e2e = compute_e2e_delay(placement, routes, slice_req, substrate)
    if e2e > slice_req.qos.max_e2e_delay:
        return False, "c7_delay"

    return True, None


def classify_kill(slice_req, substrate):
    """Classify a structural kill with FULL verifier-equivalent checks.

    Returns: bin_name, detail_string
    """
    feasible_nodes = get_feasible_nodes_per_vnf(slice_req, substrate)

    # Bin 1: PHYSICAL — at least one VNF has zero feasible nodes
    for i, nodes in enumerate(feasible_nodes):
        if len(nodes) == 0:
            return "physical", f"VNF {slice_req.vnfs[i].vnf_id} has no feasible node"

    # Enumerate all possible node assignments
    combos = list(itertools.product(*feasible_nodes))
    if len(combos) > MAX_NODE_COMBOS:
        rng = np.random.default_rng(hash(slice_req.request_id) % 2**32)
        indices = rng.choice(len(combos), MAX_NODE_COMBOS, replace=False)
        combos = [combos[i] for i in indices]

    any_placement_valid = False
    any_routing_ok = False
    fail_reasons = Counter()

    for combo in combos:
        placement = list(combo)
        feasible, reason = try_placement_full_check(placement, slice_req, substrate)

        if feasible:
            return "search_failure", "FFD missed a valid placement (passes full verifier)"

        if reason == "capacity":
            continue  # Node overload — not a valid placement
        else:
            any_placement_valid = True
            if reason in ("c5b_routing",):
                pass  # Routing failed
            else:
                any_routing_ok = True  # Routing succeeded but delay/hops failed
            fail_reasons[reason] += 1

    if not any_placement_valid:
        return "physical", "no valid placement (capacity conflicts across VNFs)"

    # All placements fail post-routing checks
    if fail_reasons.get("c7_delay", 0) > 0 and fail_reasons.get("c5b_routing", 0) == 0:
        return "c7_trap", "placements route but all fail M/M/1 delay"
    if fail_reasons.get("c5b_routing", 0) > 0 and fail_reasons.get("c7_delay", 0) == 0:
        return "c5b_trap", "placements exist but all fail BW routing"
    if fail_reasons.get("c9_hops", 0) > 0 and fail_reasons.get("c5b_routing", 0) == 0 \
       and fail_reasons.get("c7_delay", 0) == 0:
        return "c9_trap", "placements route but all exceed hop limit"

    # Mixed failures
    dominant = fail_reasons.most_common(1)[0][0] if fail_reasons else "unknown"
    return f"{dominant}_trap", f"mixed failures: {dict(fail_reasons)}"


def run_seed(arrival_seed, substrate):
    """Run kill classification for one seed on a FRESH substrate."""
    sub = copy.deepcopy(substrate)
    rng = np.random.default_rng(arrival_seed)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    total_arrivals = 0
    kills = Counter()
    kill_details = Counter()

    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue

        total_arrivals += 1
        sr = event.slice_request

        result = _run_greedy_ffd(sub, sr, GreedyConfig())
        if result.feasible:
            continue

        bin_name, detail = classify_kill(sr, sub)
        kills[bin_name] += 1
        kill_details[detail] += 1

    total_killed = sum(kills.values())
    return total_arrivals, total_arrivals - total_killed, kills, kill_details


def main():
    sub = build_substrate(SUBSTRATE_SEED)
    g = sub.graph

    logger.info("=" * 90)
    logger.info("KILL CLASSIFICATION v2 — full verifier gate, computed bins")
    logger.info("  Delay: load-dependent M/M/1, evaluated at arrival-state substrate")
    logger.info("  Substrate: 5 domains x 3 nodes, 10%% edge/10%% MEC, seed=%d", SUBSTRATE_SEED)
    logger.info("  Arrivals: %d per seed, seeds: %s", NUM_ARRIVALS, ARRIVAL_SEEDS)
    logger.info("=" * 90)

    for dom in range(NUM_DOMAINS):
        tiers = {}
        cpu_total = 0
        for nid, d in g.nodes(data=True):
            if d.get("domain_id") == dom:
                t = d["tier"]
                tiers[t] = tiers.get(t, 0) + 1
                cpu_total += float(d["cpu_capacity"])
        tier_str = ", ".join(f"{t}:{c}" for t, c in sorted(tiers.items()))
        logger.info("  Domain %d: %s  (%.0f CPU)", dom, tier_str, cpu_total)

    all_kills = Counter()
    all_details = Counter()
    kill_rates = []
    per_seed_bins = []

    for seed in ARRIVAL_SEEDS:
        t0 = time.time()
        total, admitted, kills, details = run_seed(seed, sub)
        elapsed = time.time() - t0
        total_killed = sum(kills.values())
        kill_rate = 100 * total_killed / total if total > 0 else 0

        kill_rates.append(kill_rate)
        per_seed_bins.append(kills)
        for k, v in kills.items():
            all_kills[k] += v
        for k, v in details.items():
            all_details[k] += v

        logger.info("")
        logger.info("--- Seed %d (%.1fs) ---", seed, elapsed)
        logger.info("  Arrivals: %d, FFD admitted: %d (%.1f%%), killed: %d (%.1f%%)",
                    total, admitted, 100 * admitted / total, total_killed, kill_rate)
        for bin_name in ["physical", "search_failure", "c5b_trap", "c7_trap", "c9_trap"]:
            count = kills.get(bin_name, 0)
            if count > 0 or bin_name in ["physical", "search_failure", "c5b_trap", "c7_trap"]:
                pct = 100 * count / total_killed if total_killed > 0 else 0
                logger.info("    %-16s: %3d  (%.1f%% of kills)", bin_name, count, pct)
        # Any mixed/other bins
        for bin_name in sorted(kills.keys()):
            if bin_name not in ["physical", "search_failure", "c5b_trap", "c7_trap", "c9_trap"]:
                count = kills[bin_name]
                pct = 100 * count / total_killed if total_killed > 0 else 0
                logger.info("    %-16s: %3d  (%.1f%% of kills)", bin_name, count, pct)

    total_kills = sum(all_kills.values())
    total_arrivals_all = NUM_ARRIVALS * len(ARRIVAL_SEEDS)

    logger.info("")
    logger.info("=" * 90)
    logger.info("AGGREGATE (%d seeds, %d total arrivals, %d total kills)",
                len(ARRIVAL_SEEDS), total_arrivals_all, total_kills)
    logger.info("=" * 90)

    logger.info("  Kill rate: %.1f%% +/- %.1f%%", np.mean(kill_rates), np.std(kill_rates))
    logger.info("")

    all_bins = ["physical", "search_failure", "c5b_trap", "c7_trap", "c9_trap"]
    # Add any extra bins
    for k in sorted(all_kills.keys()):
        if k not in all_bins:
            all_bins.append(k)

    for bin_name in all_bins:
        count = all_kills.get(bin_name, 0)
        if count == 0 and bin_name not in ["physical", "search_failure", "c5b_trap", "c7_trap"]:
            continue
        pct = 100 * count / total_kills if total_kills > 0 else 0
        per_seed = [s.get(bin_name, 0) for s in per_seed_bins]
        logger.info("  %-16s: %4d / %d  (%.1f%% of kills)  per-seed: %s",
                    bin_name, count, total_kills, pct, per_seed)

    logger.info("")
    logger.info("DETAIL BREAKDOWN:")
    for detail, count in all_details.most_common():
        logger.info("  %s: %d", detail, count)

    logger.info("")
    logger.info("=" * 90)
    logger.info("VERDICT")
    logger.info("=" * 90)

    phys_pct = 100 * all_kills.get("physical", 0) / total_kills if total_kills > 0 else 0
    search_pct = 100 * all_kills.get("search_failure", 0) / total_kills if total_kills > 0 else 0
    c5b_pct = 100 * all_kills.get("c5b_trap", 0) / total_kills if total_kills > 0 else 0
    c7_pct = 100 * all_kills.get("c7_trap", 0) / total_kills if total_kills > 0 else 0

    logger.info("  Physical:       %.1f%%", phys_pct)
    logger.info("  Search failure: %.1f%%  (FFD missed valid placements)", search_pct)
    logger.info("  C5b trap:       %.1f%%  (placements exist, routing fails)", c5b_pct)
    logger.info("  C7 trap:        %.1f%%  (routes exist, M/M/1 delay fails)", c7_pct)

    if search_pct > 30:
        logger.info("")
        logger.info("  SEARCH FAILURES DOMINATE. Plan builder is the headline.")
    elif phys_pct > 70:
        logger.info("")
        logger.info("  PHYSICAL KILLS DOMINATE. Topology-inherent, fix requires distribution.")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
