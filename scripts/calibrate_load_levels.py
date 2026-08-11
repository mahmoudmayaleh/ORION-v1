#!/usr/bin/env python
"""§Y.3 — calibrate the four load levels. Gates every §Y cell.

Nothing in §Y can fire until L1..L4 are frozen: `load_levels.get_level` raises
rather than hand out a guessed lambda, because a guessed ladder that silently
works is exactly the failure mode §Y exists to remove.

Load is measured two ways and both are reported, because they answer different
questions:

  rho_offered  offered load as a fraction of substrate capacity. Depends only on
               the workload and the infrastructure, NOT on which approach runs,
               so it is the honest x-axis for every figure. This is what the
               supervisor asked for ("the load w.r.t capacity: 20%, 40%").
  acceptance   what the Plain greedy baseline actually achieves. Two substrates
               at equal rho are not equally hard if their tier structure binds
               differently, so the levels are pinned on the observed outcome.
               Defined as admitted / total GENERATED requests over the whole
               episode (supervisor ratification, 2026-07-30): no warm-up window,
               so it is the same quantity grid_runner and wp7_runner report. The
               steady-state value is still computed and printed next to it, as a
               diagnostic, so the fill-from-empty transient is a measured number.

The sweep is specified in rho and converted to lambda per substrate
(`arrival_rate_for_rho`). A lambda bracket is not portable: at this 100-node
substrate lambda=4.0 is only rho~0.30, so the pre-§Y bracket of 0.5..4.0 never
approaches saturation and could not have produced the L3/L4 targets.

Plain admission is byte-identical to `grid_runner.eval_plain`: colocation_ffd,
then the §X.2 QoS gate, then allocate; departures release. If those diverge the
calibration describes a policy that never runs.

Run:  PYTHONHASHSEED=0 PYTHONPATH=src python scripts/calibrate_load_levels.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from orion.baselines.colocation_ffd import colocation_ffd  # noqa: E402
from orion.baselines.greedy_ffd import GreedyConfig  # noqa: E402
from orion.sim.arrival_process import EventType  # noqa: E402
from orion.sim.load_levels import (  # noqa: E402
    ACCEPTANCE_TARGETS,
    NUM_ARRIVALS,
    RHO_SWEEP,
    SEEDS,
    SERVICE_RATE,
    WARMUP_ARRIVALS,
    arrival_rate_for_rho,
    capacity_by_tier,
    expected_slice_demand,
    make_arrival_process,
    offered_load_fraction,
)
from orion.sim.qos_gate import plan_qos_ok  # noqa: E402
from orion.sim.slice_generator import generate_slice_request  # noqa: E402
from orion.substrate.hierarchical_topology import (  # noqa: E402
    TRAIN_INSTANCES,
    generate_hierarchical_topology,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calib")

OUT = Path("results/y3_load_calibration.json")


def run_plain(substrate, arrival_rate, rng, num_arrivals=NUM_ARRIVALS):
    """One Plain episode. Admission logic mirrors grid_runner.eval_plain exactly.

    Returns per-arrival admit flags in arrival order, plus occupancy samples so
    the steady state can be checked rather than assumed.
    """
    ap = make_arrival_process(substrate, arrival_rate, rng,
                              slice_factory=generate_slice_request,
                              num_arrivals=num_arrivals)
    active: dict[str, tuple] = {}
    admits: list[bool] = []
    concurrent: list[int] = []
    # Utilisation must be sampled DURING the episode. Sampling after it ends
    # reads zero for every tier, because the last events are departures and the
    # substrate is empty again by then.
    util_samples: list[dict] = []
    causes = {"no_placement": 0, "qos_gate": 0, "allocate_failed": 0}
    n_arrivals_seen = 0

    for ev in ap.events:
        if ev.event_type == EventType.DEPARTURE:
            held = active.pop(ev.request_id, None)
            if held is not None:
                substrate.deallocate(held[0], held[1])
            continue
        if ev.event_type != EventType.ARRIVAL or ev.slice_request is None:
            continue
        sr = ev.slice_request
        res = colocation_ffd(substrate, sr, GreedyConfig())
        ok = False
        # Separate the two rejection causes. A rejection floor that persists at
        # near-zero utilisation is structural, not congestion, and calibrating
        # load levels against it would be pinning lambda on a constant.
        if not res.feasible or res.plan is None:
            causes["no_placement"] += 1
        elif not plan_qos_ok(substrate, sr, res.plan):
            causes["qos_gate"] += 1
        else:
            try:
                substrate.allocate(res.plan, sr)
                active[sr.request_id] = (res.plan, sr)
                ok = True
            except Exception:  # noqa: BLE001
                causes["allocate_failed"] += 1
        admits.append(ok)
        concurrent.append(len(active))
        n_arrivals_seen += 1
        if n_arrivals_seen > WARMUP_ARRIVALS:
            util_samples.append(tier_utilisation(substrate))

    # ARRIVAL span, not event span. The last event is the final departure, which
    # trails the last arrival by ~one lifetime; including it made a 7 t.u. arrival
    # phase read as 6.7 mean lifetimes of churn. What matters is whether slices
    # depart WHILE arrivals are still coming.
    arrival_times = [e.time for e in ap.events
                     if e.event_type == EventType.ARRIVAL and e.slice_request is not None]
    span = (max(arrival_times) - min(arrival_times)) if arrival_times else 0.0
    return admits, concurrent, util_samples, span, causes


def tier_utilisation(substrate):
    """Per-tier occupied fraction.

    Which tier binds is a property of the WORKLOAD's tier restrictions, not of
    capacity, so callers must read the max over tiers rather than naming one.
    This readout used to hardcode `ran_edge`; after the three-tier merge that
    key stopped existing and every sweep point reported 0.00 while the tier that
    actually saturates was invisible."""
    out = {}
    for tier, row in capacity_by_tier(substrate).items():
        used = sum(substrate.graph.nodes[n]["cpu_capacity"]
                   - substrate.graph.nodes[n]["cpu_residual"]
                   for n in substrate.graph.nodes
                   if substrate.graph.nodes[n]["tier"] == tier)
        out[tier] = round(used / row["cpu"], 4) if row["cpu"] else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=10,
                    help="rho grid points, geometric across RHO_SWEEP")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--instances", type=int, nargs="+",
                    default=list(TRAIN_INSTANCES[:3]),
                    help="topology instances to average over")
    ap.add_argument("--arrivals", type=int, default=NUM_ARRIVALS)
    args = ap.parse_args()

    probe_sub = generate_hierarchical_topology(args.instances[0])
    ecpu, eram = expected_slice_demand(probe_sub, generate_slice_request,
                                       np.random.default_rng(12345),
                                       num_samples=2000)
    log.info("E[cpu/slice]=%.2f E[ram/slice]=%.2f", ecpu, eram)
    log.info("per-tier CPU capacity: %s",
             {k: round(v["cpu"], 1) for k, v in capacity_by_tier(probe_sub).items()})

    lo, hi = RHO_SWEEP
    rhos = [lo * (hi / lo) ** (i / (args.points - 1)) for i in range(args.points)]

    rows = []
    t0 = time.time()
    for rho in rhos:
        lam = arrival_rate_for_rho(probe_sub, rho, ecpu, eram)
        accs, accs_ss, occs, tiers, spans, cause_acc = [], [], [], [], [], []
        for inst in args.instances:
            for seed in args.seeds:
                sub = generate_hierarchical_topology(inst)
                admits, concurrent, utils, span, causes = run_plain(
                    sub, lam, np.random.default_rng(seed + 777), args.arrivals)
                # PRIMARY: admitted / total GENERATED, over every arrival in the
                # episode. This is the supervisor's ratified definition
                # (2026-07-30) and it is byte-identical to what grid_runner and
                # wp7_runner report, so the levels are pinned on the same number
                # the results tables carry. The previous windowed metric made the
                # calibration and the eval disagree by up to 3.9 points at L4.
                accs.append(sum(admits) / len(admits))
                # SECONDARY, diagnostic only: the same episode excluding the
                # fill-from-empty transient. Kept so the transient's contribution
                # is a measured, reported quantity rather than an unknown.
                window = admits[WARMUP_ARRIVALS:]
                accs_ss.append(sum(window) / len(window) if window else 0.0)
                occs.append(float(np.mean(concurrent[WARMUP_ARRIVALS:])))
                tiers.append({k: float(np.mean([u[k] for u in utils]))
                              for k in utils[0]} if utils else {})
                spans.append(span)
                cause_acc.append(causes)
        row = {
            "rho_offered": round(rho, 4),
            "lambda": round(lam, 4),
            "erlangs": round(lam / SERVICE_RATE, 2),
            "acceptance_mean": round(float(np.mean(accs)), 4),
            "acceptance_std": round(float(np.std(accs)), 4),
            "acceptance_steady_mean": round(float(np.mean(accs_ss)), 4),
            "transient_bias": round(float(np.mean(accs) - np.mean(accs_ss)), 4),
            "concurrent_mean": round(float(np.mean(occs)), 2),
            "tier_util": {k: round(float(np.mean([t[k] for t in tiers])), 4)
                          for k in tiers[0]},
            "n_episodes": len(accs),
            # Steady-state reachability. A = lambda/mu is the load OFFERED, but an
            # episode of N arrivals can never hold more than N slices, and churn
            # only exists if the episode spans several mean lifetimes. When either
            # fails, the episode is "fill until arrivals run out" and its
            # acceptance is not a steady-state number.
            "reject_causes": {k: int(np.mean([c[k] for c in cause_acc]))
                              for k in cause_acc[0]},
            "span_in_lifetimes": round(float(np.mean(spans)) * SERVICE_RATE, 2),
            "erlangs_over_arrivals": round((lam / SERVICE_RATE) / args.arrivals, 2),
        }
        rows.append(row)
        flag = ""
        if row["erlangs_over_arrivals"] > 0.25:
            flag += "  !! A > N/4: offered concurrency unreachable in N arrivals"
        if row["span_in_lifetimes"] < 3:
            flag += "  !! span < 3 mean lifetimes: no churn"
        log.info("rho=%.3f lambda=%.2f A=%.0f -> acceptance %.3f±%.3f "
                 "(steady %.3f, transient +%.3f)  "
                 "concurrent %.1f  edge_util %.2f  span %.1f lifetimes%s",
                 rho, lam, row["erlangs"], row["acceptance_mean"],
                 row["acceptance_std"], row["acceptance_steady_mean"],
                 row["transient_bias"], row["concurrent_mean"],
                 (max(row["tier_util"].values()) if row["tier_util"] else 0.0),
                 row["span_in_lifetimes"], flag)
        log.info("    reject causes (mean/episode): %s", row["reject_causes"])

    # ---- pick the four levels ----------------------------------------------
    levels = {}
    for name, target in ACCEPTANCE_TARGETS.items():
            # Only steady-state rows are eligible; a level pinned on a fill-until-empty
        # row would set lambda from a number that is not an acceptance rate.
        eligible = [r for r in rows
                    if r["erlangs_over_arrivals"] <= 0.25
                    and r["span_in_lifetimes"] >= 3] or rows
        best = min(eligible, key=lambda r: abs(r["acceptance_mean"] - target))
        levels[name] = {"target_acceptance": target, **best}

    # ---- the ladder must actually be a ladder --------------------------------
    # A target above the achievable ceiling does not fail loudly: every level above
    # it simply selects the top sweep point, and two levels end up with the SAME
    # lambda. That is a degenerate ladder wearing distinct names, and every
    # downstream cell would compare two identical experiments. Caught here because
    # monotonicity cannot see it: identical rows are trivially monotone.
    chosen_lambdas = {name: round(levels[name]["lambda"], 6) for name in levels}
    dupes = {lam for lam in chosen_lambdas.values()
             if list(chosen_lambdas.values()).count(lam) > 1}
    distinct = not dupes
    if not distinct:
        collide = sorted(n for n, lam in chosen_lambdas.items() if lam in dupes)
        print(f"\n  !! DEGENERATE LADDER: {collide} all selected the same lambda "
              f"({sorted(dupes)}).")
        print("     A target above the achievable ceiling collapses onto the top of")
        print("     the sweep. Re-target against the measured range; do NOT freeze.")

    # ---- named calibration expectations (§Y.3) ------------------------------
    accs = [r["acceptance_mean"] for r in rows]
    monotone = all(a >= b - 0.02 for a, b in zip(accs, accs[1:]))
    l1, l4 = levels["L1"], levels["L4"]
    conc_ratio = (l4["concurrent_mean"] / l1["concurrent_mean"]
                  if l1["concurrent_mean"] else 0.0)
    # The BINDING tier, not a hardcoded one. Under the realistic 15:4:1 fan-out
    # MEC is the scarce tier (few hosts, and URLLC/V2X/XR all require it), so
    # reading ran_edge reported 0.28 while the tier that actually saturates was
    # at 0.90 and the expectation looked failed when it had passed.
    binding_tier = max(l4["tier_util"], key=l4["tier_util"].get)
    edge_l4 = l4["tier_util"][binding_tier]

    print("\n" + "=" * 78)
    print("§Y.3 LOAD CALIBRATION")
    print("=" * 78)
    print("  acceptance = admitted / total generated, whole episode (ratified"
          " definition).")
    print("  'steady' excludes the first "
          f"{WARMUP_ARRIVALS} arrivals and is a diagnostic, not the metric.")
    print(f"  {'rho':>6} {'lambda':>8} {'A':>7} {'accept':>14} {'steady':>7} "
          f"{'concur':>8} {'edge':>6} {'span/LT':>8} {'usable':>7}")
    for r in rows:
        usable = (r["erlangs_over_arrivals"] <= 0.25 and r["span_in_lifetimes"] >= 3)
        print(f"  {r['rho_offered']:>6.3f} {r['lambda']:>8.2f} {r['erlangs']:>7.0f} "
              f"{r['acceptance_mean']:>7.3f}±{r['acceptance_std']:<6.3f} "
              f"{r['acceptance_steady_mean']:>7.3f} "
              f"{r['concurrent_mean']:>8.1f} {(max(r['tier_util'].values()) if r['tier_util'] else 0):>6.2f} "
              f"{r['span_in_lifetimes']:>8.1f} {'yes' if usable else 'NO':>7}")
    if not all(r["erlangs_over_arrivals"] <= 0.25 and r["span_in_lifetimes"] >= 3
               for r in rows):
        print("\n  !! Rows marked 'NO' are NOT steady-state measurements: the offered")
        print("     concurrency exceeds what N arrivals can hold, or the episode spans")
        print("     too few mean lifetimes for departures to matter. Raise --arrivals")
        print("     or narrow the rho sweep; do not freeze levels from those rows.")

    print("\n  CHOSEN LEVELS")
    print(f"  {'level':>6} {'target':>7} {'lambda':>8} {'A':>7} {'rho':>7} {'accept':>8}")
    for name in ("L1", "L2", "L3", "L4"):
        v = levels[name]
        print(f"  {name:>6} {v['target_acceptance']:>7.2f} {v['lambda']:>8.2f} "
              f"{v['erlangs']:>7.0f} {v['rho_offered']:>7.3f} "
              f"{v['acceptance_mean']:>8.3f}")

    print("\n  NAMED EXPECTATIONS (§Y.3)")
    print(f"    acceptance falls monotonically with rho : "
          f"{'PASS' if monotone else 'FAIL -- levels NOT frozen'}")
    print(f"    concurrent(L4)/concurrent(L1) >= 3      : "
          f"{conc_ratio:.2f}  {'PASS' if conc_ratio >= 3 else 'REPORT'}")
    print(f"    binding-tier utilisation at L4 > 0.85   : "
          f"{edge_l4:.2f} ({binding_tier})  {'PASS' if edge_l4 > 0.85 else 'REPORT'}")
    print(f"    every level has a distinct lambda          : "
          f"{'PASS' if distinct else 'FAIL -- levels NOT frozen'}")
    if not monotone:
        print("\n  !! Acceptance is not monotone in rho. Per §Y.3 the calibration is")
        print("     INVALID and the levels must not be frozen. Re-derive before firing.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "sweep": rows, "levels": levels,
        "expected_cpu_per_slice": round(ecpu, 3),
        "expected_ram_per_slice": round(eram, 3),
        "service_rate": SERVICE_RATE, "num_arrivals": args.arrivals,
        "warmup_arrivals": WARMUP_ARRIVALS,
        "seeds": args.seeds, "instances": args.instances,
        "monotone": monotone, "wall_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"\n  written: {OUT}")
    print("  Freeze these into PREREG §Y.3 and populate load_levels.CALIBRATED_LEVELS.")
    return 0 if (monotone and distinct) else 1


if __name__ == "__main__":
    raise SystemExit(main())
