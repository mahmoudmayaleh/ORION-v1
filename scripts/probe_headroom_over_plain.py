#!/usr/bin/env python3
"""Headroom-over-Plain — the number that decides if the partitioning claim has a
paper. Run on the two T- families (C+_T-_B-, C-_T-_B-). No training, no LLM, no MDO.

Per arrival, against the base substrate (independent per-arrival feasibility, the
same model as compute_ceiling), classified with ONE verifier (C2/C3/C5/C7/C9 via
check_and_cost) so Plain and the exhaustive rescue are apples-to-apples:

  coloc_admits    : Plain-ColocFB places it single-domain AND it verifies.
  ffd_admits      : Plain-ColocFB places it multi-domain (FFD fallback) AND verifies.
  ffd_misses      : Plain-ColocFB's placement rejects/fails verify.
    exhaustive_rescues : ffd_miss that an exhaustive-best partition ADMITS.
    physically_infeasible : ffd_miss that NO placement admits (not in the ceiling).

HEADROOM-OVER-PLAIN = exhaustive_rescues as absolute FoC points = the max the MDO
could EVER add over the deployable baseline on this family. Decision: >=~8 pts alive,
<~8 pts dead-as-primary (single-seed noise ran 8-24pp; a 3pt ceiling is unmeasurable).
"""
import argparse
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
from orion.baselines.colocation_ffd import colocation_ffd
from orion.baselines.greedy_ffd import GreedyConfig
from probe_b_partition_regime import check_and_cost, MULTI_CAP

FAM = {f.short_name: f for f in R.ALL_FAMILIES}


def plain_class(sr, sub):
    """Plain-ColocFB placement verified with the full gate.
    Returns 'coloc' | 'ffd' | 'miss'."""
    res = colocation_ffd(sub, sr, GreedyConfig())
    if not res.feasible or res.plan is None:
        return "miss"
    placement = [res.plan.vnf_placements.get(v.vnf_id) for v in sr.vnfs]
    if any(n is None for n in placement):
        return "miss"
    ok, _ = check_and_cost(placement, sr, sub)
    if not ok:
        return "miss"
    doms = {sub.graph.nodes[n]["domain_id"] for n in placement}
    return "coloc" if len(doms) == 1 else "ffd"


def any_feasible(sr, sub):
    """Exhaustive-best: does ANY placement admit (full gate)?  (rescue check)"""
    feasible = R._get_feasible_nodes(sr, sub)
    if any(len(c) == 0 for c in feasible):
        return False
    n = math.prod(len(c) for c in feasible)
    if n <= MULTI_CAP:
        combos = itertools.product(*feasible)
    else:
        rng = np.random.default_rng(hash(sr.request_id) % 2**32)
        combos = ([feasible[j][rng.integers(len(feasible[j]))] for j in range(len(feasible))]
                  for _ in range(MULTI_CAP))
    for combo in combos:
        ok, _ = check_and_cost(list(combo), sr, sub)
        if ok:
            return True
    return False


def run_stream(fname, iseed, arrival_seed):
    sub = R.generate_family_instance(FAM[fname], seed=iseed)
    rng = np.random.default_rng(arrival_seed)
    ap = R.ArrivalProcess(sub, R.ARRIVALS_PER_INSTANCE, R.ARRIVAL_RATE, R.SERVICE_RATE, rng)
    ap.generate()
    c = defaultdict(int)
    for ev in ap.events:
        if ev.event_type != R.EventType.ARRIVAL or ev.slice_request is None:
            continue
        c["total"] += 1
        cls = plain_class(ev.slice_request, sub)
        if cls == "coloc":
            c["coloc_admits"] += 1
        elif cls == "ffd":
            c["ffd_admits"] += 1
        else:
            c["ffd_misses"] += 1
            if any_feasible(ev.slice_request, sub):
                c["exhaustive_rescues"] += 1
            else:
                c["physically_infeasible"] += 1
    return c


def agg_family(fname, iseeds, arrival_seeds):
    tot = defaultdict(int)
    for iseed in iseeds:
        for aseed in arrival_seeds:
            c = run_stream(fname, iseed, aseed)
            for k, v in c.items():
                tot[k] += v
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrival-seeds", nargs="+", type=int, default=R.RUN_SEEDS)
    args = ap.parse_args()

    # C-_T-_B- uses TRAIN instance seeds; C+_T-_B- is held-out (TEST instance seeds).
    families = [("C+_T-_B-", R.TEST_INSTANCE_SEEDS), ("C-_T-_B-", R.TRAIN_INSTANCE_SEEDS)]

    print(f"Headroom-over-Plain  (arrival_seeds={args.arrival_seeds})\n")
    for fname, iseeds in families:
        c = agg_family(fname, iseeds, args.arrival_seeds)
        total = c["total"]
        plain_admits = c["coloc_admits"] + c["ffd_admits"]
        rescues = c["exhaustive_rescues"]
        ceiling = plain_admits + rescues
        foc_plain = 100 * plain_admits / ceiling if ceiling else 0.0
        headroom_foc = 100 * rescues / ceiling if ceiling else 0.0   # FoC points
        headroom_stream = 100 * rescues / total if total else 0.0    # % of stream
        misses = c["ffd_misses"]
        resc_frac = 100 * rescues / misses if misses else 0.0
        inf_frac = 100 * c["physically_infeasible"] / misses if misses else 0.0

        print(f"=== {fname}  (iseeds={iseeds}, n={total}) ===")
        print(f"  coloc_admits={c['coloc_admits']}  ffd_admits={c['ffd_admits']}  "
              f"ffd_misses={misses}  (rescuable={rescues}, phys_infeasible={c['physically_infeasible']})")
        print(f"  Plain-ColocFB FoC       = {foc_plain:.1f}%  (admits {plain_admits}/{ceiling} ceiling)")
        print(f"  ffd_misses breakdown    : {resc_frac:.0f}% rescuable, {inf_frac:.0f}% physically infeasible")
        print(f"  >>> HEADROOM-OVER-PLAIN = {headroom_foc:.1f} FoC pts  "
              f"({rescues} rescues; {headroom_stream:.1f}% of stream)")
        verdict = "ALIVE (>=~8 pts) -> headline target" if headroom_foc >= 8 else \
                  "DEAD as PRIMARY (<~8 pts; below single-seed noise) -> pivot headline"
        print(f"  VERDICT: {verdict}\n")


if __name__ == "__main__":
    main()
