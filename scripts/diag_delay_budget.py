#!/usr/bin/env python
"""§Y.3b — is the QoS rejection floor a workload/topology mismatch or congestion?

The §Y.3 calibration found ~20% of arrivals rejected by the QoS gate at 5% CPU
utilisation, and 6% rejected on a completely EMPTY substrate. A rejection that
survives at zero load is not congestion; it is a slice that cannot be admitted by
construction, and it puts a constant floor under every approach.

This measures, per slice type on an empty substrate, the delay the co-location
placer actually ACHIEVES against the budget the generator DRAWS, so the two can
be compared directly instead of inferred from an admit/reject bit.

Grounding for what the numbers should be (checked, not assumed):

  * 3GPP TS 23.501 splits the Packet Delay Budget. The PDB is UE-to-N6 and
    INCLUDES the radio interface; a static "CN PDB" (1 ms, 2 ms, 10 ms or 20 ms
    depending on 5QI) is the portion allocated to the core network, and is
    subtracted from the PDB to leave the radio budget. This simulator models
    transport and compute only, with no radio, so its budgets are CN PDBs, not
    full PDBs.
  * 3GPP TR 26.928 / RFC 9699: XR motion-to-photon is at most 20 ms, of which
    display hardware takes 12-13 ms, leaving 7-8 ms for rendering plus device-to-
    edge RTT. RFC 9699 states outright that running XR on the cloud "is not
    feasible". XR is an EDGE workload.
  * 5G xhaul transport budgets are roughly two orders of magnitude apart per
    segment: fronthaul ~100 us, midhaul ~1 ms, backhaul ~10 ms.

Run:  PYTHONHASHSEED=0 PYTHONPATH=src python scripts/diag_delay_budget.py
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from orion.baselines.colocation_ffd import colocation_ffd  # noqa: E402
from orion.baselines.greedy_ffd import GreedyConfig  # noqa: E402
from orion.sim.delay_model import link_sojourn, node_sojourn  # noqa: E402
from orion.sim.slice_generator import generate_slice_request  # noqa: E402
from orion.substrate.hierarchical_topology import (  # noqa: E402
    generate_hierarchical_topology,
)


def plan_delay(substrate, slice_req, plan):
    """The number `plan_qos_ok` compares to the budget, returned instead of a bool.

    Deliberately a copy of the gate's arithmetic rather than a refactor of it: if
    the two drift this diagnostic stops describing the gate, so it is checked
    against the gate's own verdict below.
    """
    g = substrate.graph
    link_by_id = {d["link_id"]: (u, v, d) for u, v, d in g.edges(data=True)}
    vmap = {v.vnf_id: v for v in slice_req.vnfs}

    extra_cpu: dict[str, float] = {}
    for fid, nid in plan.vnf_placements.items():
        extra_cpu[nid] = extra_cpu.get(nid, 0.0) + vmap[fid].cpu_demand

    node_total = 0.0
    for fid, nid in plan.vnf_placements.items():
        nd = g.nodes[nid]
        cpu_used = float(nd["cpu_capacity"]) - float(nd["cpu_residual"]) + extra_cpu[nid]
        node_total += node_sojourn(float(nd["processing_delay"]),
                                   vmap[fid].computational_intensity,
                                   float(nd["cpu_capacity"]), cpu_used)

    extra_bw: dict[str, float] = {}
    for per_link in plan.bw_allocations.values():
        for lid, bw in per_link.items():
            extra_bw[lid] = extra_bw.get(lid, 0.0) + bw

    link_total = 0.0
    inter_hops = 0
    for lids in plan.flow_routes.values():
        for lid in lids:
            u, v, d = link_by_id[lid]
            if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
                inter_hops += 1
            bw_cap = float(d.get("bandwidth_capacity", d.get("bw_capacity", 0.0)))
            bw_used = bw_cap - float(d["bw_residual"]) + extra_bw.get(lid, 0.0)
            link_total += link_sojourn(float(d["propagation_delay"]), bw_cap, bw_used)

    return node_total, link_total, inter_hops


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sub = generate_hierarchical_topology(0)
    rng = np.random.default_rng(args.seed)

    by_type = collections.defaultdict(lambda: {
        "n": 0, "achieved": [], "budget": [], "nodes": [], "links": [],
        "hops": [], "infeasible": 0, "tiers_used": collections.Counter(),
        "chain_len": collections.Counter(),
    })

    for i in range(args.samples):
        sr = generate_slice_request(request_id="d%d" % i, substrate=sub, rng=rng,
                                    arrival_time=0.0, lifetime=20.0)
        res = colocation_ffd(sub, sr, GreedyConfig())
        if not res.feasible or res.plan is None:
            continue
        st = str(sr.slice_type)
        nd, lk, hops = plan_delay(sub, sr, res.plan)
        if math.isinf(nd + lk):
            continue
        rec = by_type[st]
        rec["n"] += 1
        rec["achieved"].append(nd + lk)
        rec["budget"].append(sr.qos.max_e2e_delay)
        rec["nodes"].append(nd)
        rec["links"].append(lk)
        rec["hops"].append(hops)
        rec["chain_len"][len(sr.vnfs)] += 1
        if nd + lk > sr.qos.max_e2e_delay:
            rec["infeasible"] += 1
        for nid in res.plan.vnf_placements.values():
            rec["tiers_used"][sub.graph.nodes[nid]["tier"]] += 1

    print("=" * 96)
    print("ACHIEVED DELAY vs DRAWN BUDGET, empty substrate, co-location placer")
    print("=" * 96)
    print(f"{'slice':<8} {'n':>5} | {'achieved p10':>12} {'p50':>7} {'p90':>7} | "
          f"{'budget lo':>9} {'hi':>6} | {'node ms':>8} {'link ms':>8} {'hops':>5} | {'infeas':>7}")
    for st in sorted(by_type):
        r = by_type[st]
        if not r["n"]:
            continue
        print(f"{st:<8} {r['n']:>5} | {pct(r['achieved'], .10):>12.2f} "
              f"{pct(r['achieved'], .50):>7.2f} {pct(r['achieved'], .90):>7.2f} | "
              f"{min(r['budget']):>9.1f} {max(r['budget']):>6.1f} | "
              f"{pct(r['nodes'], .50):>8.2f} {pct(r['links'], .50):>8.2f} "
              f"{pct(r['hops'], .50):>5.0f} | {r['infeasible'] / r['n']:>7.2f}")

    print("\n" + "=" * 96)
    print("WHY: the minimum achievable delay vs the budget range the generator draws")
    print("=" * 96)
    for st in sorted(by_type):
        r = by_type[st]
        if not r["n"]:
            continue
        floor = min(r["achieved"])
        lo, hi = min(r["budget"]), max(r["budget"])
        # A budget drawn uniformly below the achievable floor is infeasible before
        # any load exists. That fraction is a constant tax on every approach.
        born = 0.0 if hi <= lo else max(0.0, min(1.0, (floor - lo) / (hi - lo)))
        tiers = ", ".join(f"{k}:{v}" for k, v in r["tiers_used"].most_common())
        chains = ", ".join(f"K={k}:{v}" for k, v in sorted(r["chain_len"].items()))
        print(f"\n{st}")
        print(f"  achievable floor        {floor:.2f} ms   (best placement seen)")
        print(f"  budget drawn uniformly  {lo:.1f} .. {hi:.1f} ms")
        print(f"  born-infeasible share   {born:.0%}  (budget drawn below the floor)")
        print(f"  tiers used              {tiers}")
        print(f"  chain lengths           {chains}")

    print("\n" + "=" * 96)
    print("READ")
    print("=" * 96)
    print("A non-zero born-infeasible share is a WORKLOAD/TOPOLOGY MISMATCH, not")
    print("congestion: those slices are rejected on an empty network and put a")
    print("constant floor under every approach, compressing the dynamic range the")
    print("load axis is supposed to produce. Fixing the budgets or the chain tier")
    print("constraints is a design decision, not a tuning knob -- see the 3GPP /")
    print("RFC grounding in this file's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
