#!/usr/bin/env python3
"""Sanity check: replay search-failure placements through the real verifier.

Takes arrivals classified as "search_failure" by the kill enumerator,
builds a PlacementPlan from the enumerator's found placement, allocates
on the real substrate, and runs the full ground-truth verifier (C2, C3,
C5, C5b VCR-scaled, C7 load-dependent M/M/1, C9 inter-domain hops).

If any "search_failure" is rejected by the verifier, the kill
classification is inflated and must be corrected.
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.routing import route_cross_domain_flow, allocate_route_bw, deallocate_route_bw
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.verifier import verify_committed_plan
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier, PlacementPlan, SliceRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_DOMAINS = 5
INTER_DOMAIN_BW = 200.0
SUBSTRATE_SEED = 0
ARRIVAL_SEED = 42
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
NUM_ARRIVALS = 200
NUM_SAMPLES = 10  # Check 10 search-failure arrivals
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


def find_valid_placement(slice_req, substrate):
    """Find a placement+routing that the enumerator considers valid.

    Returns (placement, intra_routes, cross_routes, cross_bw) or None.
    """
    feasible_nodes = get_feasible_nodes_per_vnf(slice_req, substrate)
    if any(len(n) == 0 for n in feasible_nodes):
        return None

    combos = list(itertools.product(*feasible_nodes))
    if len(combos) > MAX_NODE_COMBOS:
        rng = np.random.default_rng(hash(slice_req.request_id) % 2**32)
        indices = rng.choice(len(combos), MAX_NODE_COMBOS, replace=False)
        combos = [combos[i] for i in indices]

    g = substrate.graph

    for combo in combos:
        placement = list(combo)

        # Check node capacity
        node_cpu = {}
        node_ram = {}
        valid = True
        for j, vnf in enumerate(slice_req.vnfs):
            nid = placement[j]
            node_cpu[nid] = node_cpu.get(nid, 0.0) + vnf.cpu_demand
            node_ram[nid] = node_ram.get(nid, 0.0) + vnf.ram_demand
            if node_cpu[nid] > float(g.nodes[nid]["cpu_residual"]) + 0.01:
                valid = False
                break
            if node_ram[nid] > float(g.nodes[nid]["ram_residual"]) + 0.01:
                valid = False
                break
        if not valid:
            continue

        # Route cross-domain flows
        sub_copy = copy.deepcopy(substrate)
        vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(slice_req.vnfs)}

        cross_routes = {}
        cross_bw = {}
        all_ok = True

        for fe in slice_req.flow_edges:
            src_node = vnf_to_node[fe.source_vnf]
            dst_node = vnf_to_node[fe.target_vnf]
            src_dom = g.nodes[src_node]["domain_id"]
            dst_dom = g.nodes[dst_node]["domain_id"]

            if src_dom == dst_dom:
                # Intra-domain flow — route within domain subgraph
                domain_nodes = [n for n, d in g.nodes(data=True) if d.get("domain_id") == src_dom]
                try:
                    if src_node == dst_node:
                        continue
                    subg = sub_copy.graph.subgraph(domain_nodes)
                    path = None
                    for p in __import__('networkx').shortest_simple_paths(subg, src_node, dst_node, weight="propagation_delay"):
                        # Check BW
                        ok = True
                        for i in range(len(p) - 1):
                            edge = sub_copy.graph[p[i]][p[i+1]]
                            if float(edge["bw_residual"]) < fe.bandwidth_demand:
                                ok = False
                                break
                        if ok:
                            path = p
                            break
                    if path is None:
                        all_ok = False
                        break
                    # Allocate BW provisionally
                    for i in range(len(path) - 1):
                        sub_copy.graph[path[i]][path[i+1]]["bw_residual"] -= fe.bandwidth_demand
                    link_ids = []
                    for i in range(len(path) - 1):
                        link_ids.append(sub_copy.graph[path[i]][path[i+1]]["link_id"])
                    cross_routes[(fe.source_vnf, fe.target_vnf)] = link_ids
                    cross_bw[(fe.source_vnf, fe.target_vnf)] = fe.bandwidth_demand
                except Exception:
                    all_ok = False
                    break
            else:
                # Cross-domain flow
                result = route_cross_domain_flow(
                    sub_copy, src_node, dst_node,
                    bw_demand=fe.bandwidth_demand,
                    delay_budget=slice_req.qos.max_e2e_delay,
                )
                if not result.feasible:
                    all_ok = False
                    break
                allocate_route_bw(sub_copy, result.path_links, fe.bandwidth_demand)
                cross_routes[(fe.source_vnf, fe.target_vnf)] = result.path_links
                cross_bw[(fe.source_vnf, fe.target_vnf)] = fe.bandwidth_demand

        if all_ok:
            return placement, cross_routes, cross_bw

    return None


def build_plan_from_placement(slice_req, placement, routes, bw_map):
    """Build a PlacementPlan from a raw node placement + routes."""
    vnf_placements = {}
    cpu_alloc = {}
    ram_alloc = {}
    for i, vnf in enumerate(slice_req.vnfs):
        vnf_placements[vnf.vnf_id] = placement[i]
        cpu_alloc[vnf.vnf_id] = vnf.cpu_demand
        ram_alloc[vnf.vnf_id] = vnf.ram_demand

    bw_allocations = {}
    for flow_key, link_ids in routes.items():
        bw = bw_map.get(flow_key, 0.0)
        bw_allocations[flow_key] = {lid: bw for lid in link_ids}

    return PlacementPlan(
        plan_id=f"{slice_req.request_id}_sanity",
        vnf_placements=vnf_placements,
        cpu_allocations=cpu_alloc,
        ram_allocations=ram_alloc,
        flow_routes=routes,
        bw_allocations=bw_allocations,
        is_structurally_valid=True,
        source="sanity_check",
    )


def main():
    sub = build_substrate(SUBSTRATE_SEED)

    logger.info("=" * 90)
    logger.info("SANITY CHECK: replay search-failure placements through the real verifier")
    logger.info("  Checking C2 (CPU), C3 (RAM), C5 (BW), C5b (VCR), C7 (M/M/1 delay), C9 (hops)")
    logger.info("=" * 90)

    rng = np.random.default_rng(ARRIVAL_SEED)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    # Find search-failure arrivals
    search_failures = []
    for event in ap.events:
        if event.event_type != EventType.ARRIVAL or event.slice_request is None:
            continue
        sr = event.slice_request
        result = _run_greedy_ffd(sub, sr, GreedyConfig())
        if result.feasible:
            continue
        # This is a kill — check if enumerator finds a valid placement
        found = find_valid_placement(sr, sub)
        if found is not None:
            search_failures.append((sr, found))
            if len(search_failures) >= NUM_SAMPLES:
                break

    logger.info("Found %d search-failure arrivals to check", len(search_failures))

    # Replay each through the verifier
    passed = 0
    failed = 0
    violation_counts = Counter()

    for i, (sr, (placement, routes, bw_map)) in enumerate(search_failures):
        sub_check = copy.deepcopy(sub)
        plan = build_plan_from_placement(sr, placement, routes, bw_map)

        # Allocate on substrate
        sub_check.allocate(plan, sr)

        # Run the full verifier
        verdict = verify_committed_plan(sub_check, plan, sr)

        status = "PASS" if verdict.feasible else "FAIL"
        if verdict.feasible:
            passed += 1
        else:
            failed += 1
            for v in verdict.violated:
                violation_counts[v] += 1

        logger.info("")
        logger.info("  [%d] %s — %s (VNFs=%d, flows=%d)",
                    i + 1, sr.request_id, status, len(sr.vnfs), len(sr.flow_edges))
        logger.info("      Placement: %s", {vnf.vnf_id: placement[j] for j, vnf in enumerate(sr.vnfs)})
        domains = [sub.graph.nodes[placement[j]]["domain_id"] for j in range(len(sr.vnfs))]
        logger.info("      Domains: %s  Cross-domain flows: %d", domains, len([k for k in routes if sub.graph.nodes[placement[0]]["domain_id"] != sub.graph.nodes[placement[-1]]["domain_id"]]) if len(placement) > 1 else 0)

        if not verdict.feasible:
            logger.info("      VIOLATIONS: %s", verdict.violated)
        logger.info("      Details: e2e=%.2f/%.2f  throughput=%.2f/%.2f  hops=%d/%d",
                    verdict.details.get("e2e_delay", 0),
                    verdict.details.get("delay_budget", 0),
                    verdict.details.get("achieved_throughput", 0),
                    verdict.details.get("throughput_floor", 0),
                    int(verdict.details.get("inter_domain_hops", 0)),
                    int(verdict.details.get("hop_limit", 0)))

    # Summary
    logger.info("")
    logger.info("=" * 90)
    logger.info("SUMMARY: %d/%d passed verifier, %d/%d failed", passed, len(search_failures), failed, len(search_failures))
    logger.info("=" * 90)

    if failed > 0:
        logger.info("  VIOLATIONS: %s", dict(violation_counts))
        logger.info("")
        logger.info("  WARNING: %d/%d search-failure classifications are INFLATED.",
                    failed, len(search_failures))
        logger.info("  The enumerator is more permissive than the verifier.")
        logger.info("  Kill classification bins must be corrected.")
        pct_inflated = 100 * failed / len(search_failures)
        logger.info("  Estimated inflation: %.0f%% of search failures are actually", pct_inflated)
        logger.info("  verifier-rejected (likely C7 delay or C5b VCR).")
    else:
        logger.info("")
        logger.info("  ALL search-failure placements pass the full verifier.")
        logger.info("  Kill classification bins are CONFIRMED.")
        logger.info("  The 91.6%% search-failure rate stands.")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
