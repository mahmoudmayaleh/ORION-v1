#!/usr/bin/env python
"""Re-derive the A.2 spread-and-fail dose-response from an R.1/R.2 result JSON.

A.2's claim: a FLEXIBLE (co-locatable) chain can reject on `cross_domain_bw` only
if the plan spread it across domains when it did not have to. So that reject class
on flexible chains is a *deductive* measure of "spread when it should have
co-located", and its rate vs inter-domain bw is the dose-response.

Why this exists as a committed script rather than an ad-hoc read: the published
curve (11% / 73% / 100% at bw 70/90/110) was computed from the 2026-07-15 R.2
traces, and that session is unfalsifiable-provenance (untracked code, unrecorded
server incarnation). Ruling 4 (2026-07-16) requires the curve be re-derived from
R.2-prime before it is cited again. A script that runs against any result JSON
makes the re-derivation checkable and repeatable, instead of a number in prose.

Usage:
    python scripts/derive_a2.py data/r_local_results_R2PRIME.json [--approach R.2]
    python scripts/derive_a2.py data/r_local_results_R12.json --approach R.2   # validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BW_FOR_SEED = {42: 70, 43: 90, 44: 110}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def derive(path: Path, approach: str) -> list[dict]:
    d = json.loads(path.read_text())
    rows = []
    for seed in sorted(BW_FOR_SEED):
        cell = d.get("cells", {}).get(f"{approach}|{seed}")
        if cell is None:
            continue
        trace = cell.get("trace") or []
        if not trace:
            print(f"  !! {approach}|{seed}: no per-arrival trace in this JSON")
            continue
        forced = [t for t in trace if t.get("forced")]
        flex = [t for t in trace if not t.get("forced")]
        # The deductive measure: flexible chain, rejected, and the reject reason is
        # a cross-domain bandwidth violation => the planner spread a chain that had
        # a co-located option.
        flex_spread = [t for t in flex
                       if not t.get("admitted") and t.get("reject") == "cross_domain_bw"]
        rows.append({
            "seed": seed,
            "bw": BW_FOR_SEED[seed],
            "forced_admit": f"{sum(1 for t in forced if t.get('admitted'))}/{len(forced)}",
            "flex_admit": f"{sum(1 for t in flex if t.get('admitted'))}/{len(flex)}",
            "flex_n": len(flex),
            "flex_spread_n": len(flex_spread),
            "flex_spread_pct": (100.0 * len(flex_spread) / len(flex)) if flex else float("nan"),
            "admitted_mean_bw_tail": _mean([t.get("bw_tail") for t in trace if t.get("admitted")]),
            "rejected_mean_bw_tail": _mean([t.get("bw_tail") for t in trace if not t.get("admitted")]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=Path)
    ap.add_argument("--approach", default="R.2")
    args = ap.parse_args()

    rows = derive(args.results_json, args.approach)
    print(f"\nA.2 spread-and-fail dose-response — {args.results_json.name}, approach {args.approach}")
    print("=" * 96)
    print(f"  {'seed':>4} {'bw':>4} {'forced admit':>13} {'flex admit':>11} "
          f"{'flex spread-and-failed':>24} {'adm bw_tail':>12} {'rej bw_tail':>12}")
    for r in rows:
        print(f"  {r['seed']:>4} {r['bw']:>4} {r['forced_admit']:>13} {r['flex_admit']:>11} "
              f"{r['flex_spread_n']:>3}/{r['flex_n']:<3} ({r['flex_spread_pct']:5.1f}%)      "
              f"{r['admitted_mean_bw_tail']:>12.1f} {r['rejected_mean_bw_tail']:>12.1f}")
    print("\n  published (phantom-session R12, R.2): 4/36 (11%) | 27/37 (73%) | 35/35 (100%)")
    if rows:
        got = " | ".join(f"{r['flex_spread_n']}/{r['flex_n']} ({r['flex_spread_pct']:.0f}%)"
                         for r in rows)
        print(f"  this run:                             {got}")
    print()


if __name__ == "__main__":
    main()
