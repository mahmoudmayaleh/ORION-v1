#!/usr/bin/env python
"""§X — MILP approach: per-arrival optimal embedding (PREREG_AMENDMENT_2026-07-26_X.md).

At each arrival of the held-out stream, solve the single-slice MILP (C1-C9,
CBC) on a residual view of the substrate; commit the returned plan to the real
substrate; departures release it. Optimal myopic policy at full observability.
NOT the offline joint optimum, and never described as one.

Timeouts and allocate failures are counted per cell and reported, not dropped.
Cells bank to data/grid_cells/ under approach "MILP"; summary + solver stats to
data/milp_approach_<TAG>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

# Behind __main__ only: fired on import it replaces ANY importer via os.execv,
# which under pytest ends the session with no traceback and rc 0. Same reasoning
# as grid_runner.py:61.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

import wp7_runner as W  # noqa: E402
import grid_runner as G  # noqa: E402
import approach_runner as F  # noqa: E402
from orion.config import MILPConfig  # noqa: E402
from orion.milp.solver import MILPSolver  # noqa: E402
from orion.sim.arrival_process import ArrivalProcess, EventType  # noqa: E402
from orion.sim.qos_gate import plan_qos_ok  # noqa: E402
from cost_metrics import CostAccumulator  # noqa: E402
from orion.sim.load_levels import NUM_ARRIVALS  # noqa: E402
from orion.substrate.hierarchical_topology import HELDOUT_INSTANCES  # noqa: E402
from orion.provenance import git_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("milp_approach")

CELLS = Path("data/grid_cells")


def residual_view(sub):
    """Copy of the substrate whose capacity fields equal current residuals.

    The MILP constraints read cpu_capacity / ram_capacity / bandwidth_capacity;
    on a loaded substrate the binding quantities are the residuals. Topology,
    tiers, delays, and link ids are unchanged, so plans parsed from the view
    apply verbatim to the real substrate.
    """
    view = copy.deepcopy(sub)
    g = view.graph
    for _, d in g.nodes(data=True):
        d["cpu_capacity"] = d["cpu_residual"]
        d["ram_capacity"] = d["ram_residual"]
    for _, _, d in g.edges(data=True):
        if "bw_residual" in d:
            d["bandwidth_capacity"] = d["bw_residual"]
    return view


def eval_milp(scenario, seed, level, instance, arrivals, time_limit):
    """One MILP cell on the §Y substrate at one calibrated load level.

    §Y.5 removed the feasibility ceiling, so this reports acceptance
    (admitted / offered) on the identical stream every other approach sees.
    """
    G._wire(scenario, level, instance)
    sub = G._substrate_fn(instance)(seed)
    rng = np.random.default_rng(seed + 777)
    ap = W._make_ap(sub, arrivals, rng)
    ap.generate()

    solver = MILPSolver(MILPConfig(time_limit=time_limit))
    active = {}
    admitted = total = alloc_fail = qos_gated = 0
    statuses: dict[str, int] = {}
    solve_times = []
    cost_acc = CostAccumulator(sub)

    for ev in ap.events:
        if ev.event_type == EventType.DEPARTURE:
            p = active.pop(ev.request_id, None)
            if p is not None:
                sub.deallocate(p[0], p[1])
            continue
        if ev.event_type != EventType.ARRIVAL or ev.slice_request is None:
            continue
        total += 1
        sr = ev.slice_request
        t0 = time.perf_counter()
        sol = solver.solve(residual_view(sub), [sr])
        solve_times.append(time.perf_counter() - t0)
        statuses[sol.status] = statuses.get(sol.status, 0) + 1
        if sol.admitted.get(sr.request_id) and sr.request_id in sol.placements:
            plan = sol.placements[sr.request_id]
            # §X.2 — the MILP's internal C7 uses the static delay model; gate
            # its plan with the verifier's sojourn model like every other approach.
            if not plan_qos_ok(sub, sr, plan):
                qos_gated += 1
                statuses["QoS-gated"] = statuses.get("QoS-gated", 0) + 1
                continue
            try:
                sub.allocate(plan, sr)
                active[sr.request_id] = (plan, sr)
                admitted += 1
                cost_acc.add_plan(sr, plan)
            except Exception as e:  # noqa: BLE001 — must be visible, never silent
                alloc_fail += 1
                log.error("ALLOCATE FAILED for %s (MILP plan vs real residuals): %s",
                          sr.request_id, e)

    st = sorted(solve_times)
    acceptance = admitted / total if total else 0.0
    return {
        "acceptance": round(acceptance, 4), "admitted": admitted, "offered": total,
        "level": level, "instance": instance,
        "statuses": statuses, "alloc_fail": alloc_fail,
        "qos_gated": qos_gated, "cost": cost_acc.summary(),
        "solve_time_mean": round(float(np.mean(st)), 3) if st else None,
        "solve_time_p95": round(float(st[int(0.95 * (len(st) - 1))]), 3) if st else None,
        "solve_time_max": round(float(st[-1]), 3) if st else None,
    }


def banked_mean(scenario, approach, fams, seeds):
    vals = []
    for fam in fams:
        for seed in seeds:
            p = CELLS / f"{scenario}_{approach}_{seed}_{fam}.json"
            if p.exists():
                vals.append(json.loads(p.read_text())["acceptance"])
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["conventional", "stress"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--arrivals", type=int, default=NUM_ARRIVALS)
    ap.add_argument("--time-limit", type=int, default=20)
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--instance", type=int, default=HELDOUT_INSTANCES[0])
    ap.add_argument("--tag", default="MILP1")
    args = ap.parse_args()

    prov = git_provenance(serving=None, tag=args.tag,
                          prereg="docs/PREREG_AMENDMENT_2026-07-26_X.md")
    log.info("provenance commit=%s dirty=%s", prov["git_commit"][:8], prov["git_dirty"])

    levels = fams = args.levels
    res = {"provenance": prov, "time_limit": args.time_limit, "cells": {}}
    out_path = Path(f"data/milp_approach_{args.tag}.json")
    CELLS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for scenario in args.scenarios:
        for seed in args.seeds:
            for fam in levels:
                cell_path = CELLS / f"{scenario}_MILP_{seed}_{fam}.json"
                if cell_path.exists():
                    cell = json.loads(cell_path.read_text())
                    log.info("cell exists, skip: %s (acceptance=%.3f)",
                             cell_path.name, cell["acceptance"])
                else:
                    cell = eval_milp(scenario, seed, fam, args.instance,
                                     args.arrivals, args.time_limit)
                    cell["provenance"] = prov
                    cell_path.write_text(json.dumps(cell, indent=2, default=str))
                    log.info("MILP %s seed=%d %s -> acceptance=%.3f (%d/%d) "
                             "solve mean %.2fs p95 %.2fs statuses=%s alloc_fail=%d",
                             scenario, seed, fam, cell["acceptance"], cell["admitted"],
                             cell["offered"],
                             cell["solve_time_mean"] or -1,
                             cell["solve_time_p95"] or -1,
                             cell["statuses"], cell["alloc_fail"])
                res["cells"][f"{scenario}|{fam}|{seed}"] = cell
                Path("data").mkdir(exist_ok=True)
                out_path.write_text(json.dumps(res, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"§X MILP APPROACH — base TEST families, seeds {args.seeds}, "
          f"time_limit={args.time_limit}s")
    print("=" * 78)
    res["summary"] = {}
    for scenario in args.scenarios:
        m = float(np.mean([c["acceptance"] for k, c in res["cells"].items()
                           if k.startswith(scenario)]))
        pl = banked_mean(scenario, "Plain", fams, args.seeds)
        pp = banked_mean(scenario, "MDO-partial", fams, args.seeds)
        xh1 = "X-H1 holds (MILP >= Plain)" if m >= pl else \
              "X-H1 VIOLATED (Plain > MILP) — explain per prereg (C7/C9, time limit)"
        print(f"  {scenario:<13} MILP mean = {m:5.1f}   Plain = {pl:5.1f}   "
              f"Plain-partial = {pp:5.1f}   {xh1}")
        res["summary"][scenario] = {"milp_mean": m, "plain_mean": pl,
                                    "plain_partial_mean": pp, "xh1": xh1}
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print(f"MILP_APPROACH_DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
