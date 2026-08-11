#!/usr/bin/env python
"""Secondary cost table — placement footprint + admission selection per approach.

Reads banked cells from data/grid_cells/ and prints, per scenario, the
per-admission secondary metrics next to FoC (which stays primary):

  FoC        primary, fraction of ceiling (context column)
  n_adm      admitted count summed over (base fams x seeds)
  demand     mean cpu+ram of ADMITTED requests  -> admission selection
  K          mean chain length of admitted      -> admission selection
  iHops      mean inter-domain link traversals  -> placement footprint
  iBW        mean inter-domain Mbps-hop         -> placement footprint

Cells banked before cost capture existed show "--"; re-running an approach's runner
(or --refresh-plain / --refresh-milp here) fills them. The refresh flags
recompute ONLY the "cost" field in place and assert the banked FoC is
reproduced, so no primary number can silently move.

Usage:
  python scripts/cost_secondary_table.py
  python scripts/cost_secondary_table.py --refresh-plain
  python scripts/cost_secondary_table.py --refresh-milp --milp-time-limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

CELLS = Path("data/grid_cells")
FAMS = ["C-_T+_B-", "C-_T+_B+", "C+_T-_B-", "C+_T+_B-"]
SEEDS = [42, 43, 44]
ARRIVALS = 100
APPROACHES = ["Plain", "MILP", "Plain-partial", "RL-alone", "RL-poprior",
        "Prior-only", "Memory-off", "Full", "Reward"]
SCENARIOS = ["conventional", "stress"]


def _refresh(scenario, approach, seed, fam, args):
    """Recompute ONLY the cost field of one Plain/MILP cell, FoC-guarded."""
    import grid_runner as G
    if approach == "Plain":
        new = G.eval_plain(scenario, seed, fam, ARRIVALS)
    else:
        from milp_approach_runner import eval_milp
        new = eval_milp(scenario, seed, fam, ARRIVALS, args.milp_time_limit)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-plain", action="store_true",
                    help="recompute cost for banked Plain cells (FoC-guarded)")
    ap.add_argument("--refresh-milp", action="store_true",
                    help="recompute cost for banked MILP cells (FoC-guarded, slow)")
    ap.add_argument("--milp-time-limit", type=int, default=20)
    args = ap.parse_args()

    refresh_approaches = ({"Plain"} if args.refresh_plain else set()) | \
                   ({"MILP"} if args.refresh_milp else set())

    for scenario in SCENARIOS:
        rows = []
        for approach in APPROACHES:
            focs, n_adm = [], 0
            agg = {"demand_mean": [], "k_mean": [], "inter_hops_mean": [],
                   "inter_bw_mean": []}
            missing = 0
            for fam in FAMS:
                for seed in SEEDS:
                    p = CELLS / f"{scenario}_{approach}_{seed}_{fam}.json"
                    if not p.exists():
                        continue
                    cell = json.loads(p.read_text())
                    cost = cell.get("cost")
                    if (not cost or "demand_mean" not in cost) and approach in refresh_approaches:
                        new = _refresh(scenario, approach, seed, fam, args)
                        if round(new["foc"], 1) != round(cell["foc"], 1):
                            raise SystemExit(
                                f"REFRESH FoC MISMATCH {p.name}: banked "
                                f"{cell['foc']} vs recomputed {new['foc']} — "
                                f"cost NOT written, investigate first.")
                        cell["cost"] = cost = new["cost"]
                        p.write_text(json.dumps(cell, indent=2, default=str))
                        print(f"  refreshed cost: {p.name}")
                    focs.append(cell["foc"])
                    if cost and cost.get("n_admitted"):
                        n_adm += cost["n_admitted"]
                        for k in agg:
                            if k in cost:
                                # weight by the cell's admission count so the
                                # pooled mean is per-admission, not per-cell
                                agg[k].append((cost[k], cost["n_admitted"]))
                    else:
                        missing += 1
            if not focs:
                continue

            def wmean(pairs):
                if not pairs:
                    return None
                w = sum(n for _, n in pairs)
                return sum(v * n for v, n in pairs) / w if w else None

            fmt = lambda v, nd=1: f"{v:.{nd}f}" if v is not None else "--"
            rows.append((approach, f"{float(np.mean(focs)):5.1f}",
                         str(n_adm) if n_adm else "--",
                         fmt(wmean(agg["demand_mean"])),
                         fmt(wmean(agg["k_mean"]), 2),
                         fmt(wmean(agg["inter_hops_mean"]), 2),
                         fmt(wmean(agg["inter_bw_mean"])),
                         f"({missing} cells no cost)" if missing else ""))
        if not rows:
            continue
        print(f"\nSECONDARY COST TABLE — {scenario}, base fams, seeds 42-44 "
              f"(admitted slices only; FoC is the primary metric)")
        print(f"{'approach':<14} {'FoC':>5} {'n_adm':>6} {'demand':>7} {'K':>5} "
              f"{'iHops':>6} {'iBW':>8}")
        for r in rows:
            print(f"{r[0]:<14} {r[1]:>5} {r[2]:>6} {r[3]:>7} {r[4]:>5} "
                  f"{r[5]:>6} {r[6]:>8}  {r[7]}")
    print("\ndemand/K = properties of the admitted REQUESTS (selection); "
          "iHops/iBW = properties of the chosen PLACEMENTS (footprint).")


if __name__ == "__main__":
    main()
