#!/usr/bin/env python3
"""Probe B — colocation-vs-partition per-slice regime check.

Gate question: is there ANY eval family where a correct multi-domain partition
BEATS colocation? If colocation is near-ceiling everywhere, the MDO's core action
(split chains across domains) has nothing to win on this eval, and full-scale +
Probe A's "fix the MDO" branch are premature (scope call, back to supervisors).

Per eval slice, on each of C-_T-_B- + the 3 held-out TEST_FAMILIES, record:
  coloc_feasible   : exists a SINGLE-domain (colocated) feasible placement
                     (checked EXHAUSTIVELY over same-domain combos -> the gate is
                     not a sampling artifact)
  any_feasible     : exists ANY feasible placement (colocated or partitioned)
  partition_needed : any_feasible AND NOT coloc_feasible   <- the money column
  partition_helps  : coloc_feasible AND best multi-domain cost < best coloc cost

No training, no LLM. One (mostly-exhaustive) placement pass per slice.
"""
import argparse
import copy
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
from orion.actors.routing import route_cross_domain_flow, allocate_route_bw
from orion.baselines.greedy_ffd import GreedyConfig
from orion.sim.delay_model import node_sojourn, link_sojourn

CFG = GreedyConfig()
MULTI_CAP = 20000  # exhaustive if combos <= this, else sample this many


def check_and_cost(placement, sr, substrate):
    """(feasible, cost) for a concrete placement under the full verifier gate
    (C2/C3 caps, per-link BW, C7 E2E delay, C9 hops). Cost uses GreedyConfig
    weights: alpha*resource + gamma_intra*intra_bw + gamma_inter*inter_bw."""
    g = substrate.graph
    node_cpu, node_ram = {}, {}
    resource_cost = 0.0
    for j, vnf in enumerate(sr.vnfs):
        nid = placement[j]
        node_cpu[nid] = node_cpu.get(nid, 0.0) + vnf.cpu_demand
        node_ram[nid] = node_ram.get(nid, 0.0) + vnf.ram_demand
        if node_cpu[nid] > float(g.nodes[nid]["cpu_residual"]) + 0.01:
            return False, math.inf
        if node_ram[nid] > float(g.nodes[nid]["ram_residual"]) + 0.01:
            return False, math.inf
        resource_cost += vnf.cpu_demand + vnf.ram_demand

    vnf_to_node = {vnf.vnf_id: placement[i] for i, vnf in enumerate(sr.vnfs)}
    sub_copy = copy.deepcopy(substrate)
    routes = {}
    for fe in sr.flow_edges:
        src, dst = vnf_to_node[fe.source_vnf], vnf_to_node[fe.target_vnf]
        if src == dst:
            continue
        src_dom, dst_dom = g.nodes[src]["domain_id"], g.nodes[dst]["domain_id"]
        if src_dom == dst_dom:
            domain_nodes = [n for n, d in g.nodes(data=True) if d.get("domain_id") == src_dom]
            subg = sub_copy.graph.subgraph(domain_nodes)
            try:
                for p in nx.shortest_simple_paths(subg, src, dst, weight="propagation_delay"):
                    if all(float(sub_copy.graph[p[i]][p[i+1]]["bw_residual"]) >= fe.bandwidth_demand
                           for i in range(len(p) - 1)):
                        for i in range(len(p) - 1):
                            sub_copy.graph[p[i]][p[i+1]]["bw_residual"] -= fe.bandwidth_demand
                        lids = [sub_copy.graph[p[i]][p[i+1]]["link_id"] for i in range(len(p) - 1)]
                        routes[(fe.source_vnf, fe.target_vnf)] = lids
                        break
                else:
                    return False, math.inf
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return False, math.inf
        else:
            result = route_cross_domain_flow(
                sub_copy, src, dst, bw_demand=fe.bandwidth_demand, delay_budget=999999.0)
            if not result.feasible:
                return False, math.inf
            allocate_route_bw(sub_copy, result.path_links, fe.bandwidth_demand)
            routes[(fe.source_vnf, fe.target_vnf)] = result.path_links

    # link_id -> (is_inter_domain) lookup
    lid_inter = {}
    for u, v, d in g.edges(data=True):
        lid_inter[d["link_id"]] = (g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"])

    hops = 0
    intra_bw = inter_bw = 0.0
    for fk, lids in routes.items():
        fe = next(f for f in sr.flow_edges if (f.source_vnf, f.target_vnf) == fk)
        for lid in lids:
            if lid_inter.get(lid, False):
                hops += 1
                inter_bw += fe.bandwidth_demand
            else:
                intra_bw += fe.bandwidth_demand
    if hops > 3:  # C9
        return False, math.inf

    # C7 E2E delay (M/M/1 sojourn)
    extra_cpu = {}
    for vnf in sr.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        extra_cpu[nid] = extra_cpu.get(nid, 0.0) + vnf.cpu_demand
    total_delay = 0.0
    for vnf in sr.vnfs:
        nid = vnf_to_node[vnf.vnf_id]
        nd = g.nodes[nid]
        cpu_used = float(nd["cpu_capacity"]) - float(nd["cpu_residual"]) + extra_cpu[nid]
        s = node_sojourn(float(nd["processing_delay"]), vnf.computational_intensity,
                         float(nd["cpu_capacity"]), cpu_used)
        if math.isinf(s):
            return False, math.inf
        total_delay += s
    extra_bw = {}
    for fe in sr.flow_edges:
        for lid in routes.get((fe.source_vnf, fe.target_vnf), []):
            extra_bw[lid] = extra_bw.get(lid, 0.0) + fe.bandwidth_demand
    for fe in sr.flow_edges:
        for lid in routes.get((fe.source_vnf, fe.target_vnf), []):
            for u, v, d in g.edges(data=True):
                if d["link_id"] == lid:
                    bw_cap = float(d.get("bandwidth_capacity", d.get("bw_capacity", 1000)))
                    bw_used = bw_cap - float(d["bw_residual"]) + extra_bw.get(lid, 0.0)
                    s = link_sojourn(float(d["propagation_delay"]), bw_cap, bw_used)
                    if math.isinf(s):
                        return False, math.inf
                    total_delay += s
                    break
    if total_delay > sr.qos.max_e2e_delay:
        return False, math.inf

    cost = (CFG.alpha * resource_cost + CFG.gamma_intra * intra_bw
            + CFG.gamma_inter * inter_bw)
    return True, cost


def classify_slice(sr, substrate):
    """-> dict with coloc_feasible/any_feasible/partition_needed/partition_helps."""
    g = substrate.graph
    feasible = R._get_feasible_nodes(sr, substrate)
    if any(len(nodes) == 0 for nodes in feasible):
        return {"coloc_feasible": False, "any_feasible": False,
                "partition_needed": False, "partition_helps": False}

    dom_of = {n: g.nodes[n]["domain_id"] for n in g.nodes}
    domains = sorted({d for d in dom_of.values() if d >= 0})

    # 1) EXHAUSTIVE colocated pass (all VNFs in one domain) -> rigorous gate.
    min_coloc = math.inf
    for d in domains:
        per_vnf = [[n for n in nodes if dom_of[n] == d] for nodes in feasible]
        if any(len(c) == 0 for c in per_vnf):
            continue
        if math.prod(len(c) for c in per_vnf) > MULTI_CAP:
            # rare: huge same-domain space; sample it
            for _ in range(MULTI_CAP):
                combo = [c[np.random.randint(len(c))] for c in per_vnf]
                ok, cost = check_and_cost(combo, sr, substrate)
                if ok:
                    min_coloc = min(min_coloc, cost)
        else:
            for combo in itertools.product(*per_vnf):
                ok, cost = check_and_cost(list(combo), sr, substrate)
                if ok:
                    min_coloc = min(min_coloc, cost)
    coloc_feasible = math.isfinite(min_coloc)

    # 2) Multi-domain pass. Full effort when NOT coloc_feasible (that's the money
    # column and must be reliable); small best-effort cap otherwise (partition_helps
    # is secondary and coloc_feasible already fixes partition_needed=False).
    cap = MULTI_CAP if not coloc_feasible else 1500
    all_combos = math.prod(len(c) for c in feasible)
    if all_combos <= cap:
        combos = itertools.product(*feasible)
    else:
        rng = np.random.default_rng(hash(sr.request_id) % 2**32)
        combos = ([feasible[j][rng.integers(len(feasible[j]))] for j in range(len(feasible))]
                  for _ in range(cap))
    min_multi = math.inf
    any_multi = False
    for combo in combos:
        combo = list(combo)
        if len({dom_of[n] for n in combo}) < 2:
            continue  # colocated; handled exhaustively above
        ok, cost = check_and_cost(combo, sr, substrate)
        if ok:
            any_multi = True
            min_multi = min(min_multi, cost)

    any_feasible = coloc_feasible or any_multi
    partition_needed = any_multi and not coloc_feasible
    partition_helps = coloc_feasible and any_multi and (min_multi < min_coloc - 1e-6)
    return {"coloc_feasible": coloc_feasible, "any_feasible": any_feasible,
            "partition_needed": partition_needed, "partition_helps": partition_helps}


def run_family(fname, iseed, arrival_seed, n_arrivals):
    sub = R.generate_family_instance({f.short_name: f for f in R.ALL_FAMILIES}[fname], seed=iseed)
    rng = np.random.default_rng(arrival_seed)
    ap = R.ArrivalProcess(sub, n_arrivals, R.ARRIVAL_RATE, R.SERVICE_RATE, rng)
    ap.generate()
    agg = defaultdict(int)
    total = 0
    for ev in ap.events:
        if ev.event_type != R.EventType.ARRIVAL or ev.slice_request is None:
            continue
        total += 1
        c = classify_slice(ev.slice_request, sub)
        for k, v in c.items():
            agg[k] += int(v)
    return total, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+",
                    default=["C-_T-_B-"] + [f.short_name for f in R.TEST_FAMILIES])
    ap.add_argument("--iseed", type=int, default=0)
    ap.add_argument("--arrival-seed", type=int, default=819)  # matches WP7 eval (seed42+777)
    ap.add_argument("--arrivals", type=int, default=100)
    args = ap.parse_args()

    print(f"Probe B: partition-vs-colocation regime  (iseed={args.iseed} "
          f"arrival_seed={args.arrival_seed} n={args.arrivals})\n")
    header = f"{'family':12s} {'n':>4s} {'coloc_feas':>11s} {'partition_NEEDED':>17s} {'part_helps_cost':>16s}"
    print(header); print("-" * len(header))
    rows = []
    for fam in args.families:
        total, agg = run_family(fam, args.iseed, args.arrival_seed, args.arrivals)
        pct = lambda k: 100 * agg[k] / total if total else 0.0
        print(f"{fam:12s} {total:4d} {pct('coloc_feasible'):10.1f}% "
              f"{pct('partition_needed'):16.1f}% {pct('partition_helps'):15.1f}%")
        rows.append((fam, total, dict(agg)))

    print("\nGATE (partition_needed = colocation FAILS but a partition SUCCEEDS):")
    mx = max((100 * r[2].get("partition_needed", 0) / r[1] if r[1] else 0) for r in rows)
    if mx < 1.0:
        print(f"  MAX partition_needed = {mx:.1f}% across all families -> NEAR-ZERO.")
        print("  Colocation is near-optimal on this eval. The MDO's core action has no")
        print("  admission headroom here. Full-scale + Probe-A 'fix the MDO' are premature;")
        print("  this is a SCOPE call (build a partition-rewarding family, or narrow the")
        print("  claim). Back to supervisors.")
    else:
        best = max(rows, key=lambda r: r[2].get("partition_needed", 0) / max(1, r[1]))
        print(f"  MAX partition_needed = {mx:.1f}%  -> family '{best[0]}' rewards partitioning.")
        print("  That family is where the MDO can demonstrably win; make it the headline")
        print("  family. Probe A tells you whether the current MDO captures that headroom.")


if __name__ == "__main__":
    main()
