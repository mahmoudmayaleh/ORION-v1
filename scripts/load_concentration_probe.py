#!/usr/bin/env python3
"""Load-concentration probe — diagnose the learned-below-random gap (gate NULL follow-up).

The gate (results/wp7/gate_colocation_C+_T-_B-.json) showed RL-alone below masked-random
even with BC actors + mu5. Offline checks eliminated reward (admission drives the gradient
~25-50x) and seed-instability (the collapsed seed is the BEST one; -8.3 is structural). The
only surviving mechanism is LOAD CONCENTRATION, which needs the per-domain selection +
conditional admit signals that were never logged. This run supplies them.

NO training change: RL-alone is trained EXACTLY as in the gate (BC actors, mu5, beta=0,
greedy m~ for obs/mask). No gate policy checkpoint was saved, so we retrain then instrument
the EVAL pass only. The `return_coord` hook on train_arm is a pure accessor.

Per seed, two EVAL passes on the byte-identical held-out stream (seed+777):
  learned : mdo_mode="deterministic"  (the trained policy's committed partitions)
  random  : mdo_mode="random"         (uniform over the SAME tier-feasible mask = the 30.5 ref)
Same coord, same BC actors, same stream -> the ONLY difference is partition SELECTION.

RAW per-arrival pairs logged (addition 2): {i, domains, admit, k, rid}. From these, offline:
  - per-domain selection count (concentration),
  - per-domain admit rate CONDITIONAL on selection  (A vs B),
  - saturation: admit rate on favored domains vs arrival index (addition 3).

PRE-REGISTERED DECISION RULE (docs, before numbers land):
  A (hotspot)        : concentration in the histogram AND conditional admits on favored
                       domains DECAY with arrival index -> fix class = exploration / load
                       feature (entropy bonus or utilization obs).
  B (uninformative o): NO concentration, or FLAT conditional admits across domains regardless
                       of selection -> o^MDO doesn't discriminate admission-relevant partition
                       differences (aggregate residuals hide per-node fit) -> fix class =
                       observation enrichment (per-domain max-single-node headroom / top-k
                       node-size summary as a domain-published scalar).
  Both present        : B first (A cannot be evaluated on an uninformative observation).
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from probe_a_plateau import build_bc_actors
from orion.mdo.coordinator import MDOConfig
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("loadconc")

FAM = {f.short_name: f for f in R.ALL_FAMILIES}
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)


def instrumented_eval(coord, family, seed, arrivals, mode):
    """One eval pass on the byte-identical stream (seed+777). Returns raw per-arrival pairs.
    domains = the FINAL committed/attempted partition (present even on reject, via the
    committed rollout transition); admit = the true MDOResult.admitted flag."""
    sub = R.generate_family_instance(FAM[family], seed=0)
    delays = W.build_delays(sub)
    rng = np.random.default_rng(seed + 777)
    ap = ArrivalProcess(sub, arrivals, W.ARRIVAL_RATE, W.SERVICE_RATE, rng)
    ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=W.greedy_plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode=mode)

    committed = [t for t in ep.rollout.mdo if getattr(t, "committed", False)]
    res = ep.mdo_results
    n = min(len(committed), len(res))
    if len(committed) != len(res):
        logger.warning("committed(%d) != mdo_results(%d); zipping min", len(committed), len(res))
    pairs = []
    for i in range(n):
        t = committed[i]; r = res[i]
        act = list(t.action) if getattr(t, "action", None) is not None else (r.partition or [])
        pairs.append({"i": i, "domains": [int(x) for x in act],
                      "admit": bool(r.admitted), "k": len(act),
                      "rid": getattr(r, "request_id", None)})
    return pairs, ep.stats.admitted, ep.stats.total_arrivals


def summarize(pairs, num_domains, label):
    """Inline A/B signatures: selection histogram, conditional admit per domain, saturation."""
    n = len(pairs)
    # per-domain: how many arrivals SELECTED d (distinct per arrival), and of those how many admit
    sel = defaultdict(int); sel_adm = defaultdict(int); place = defaultdict(int)
    for p in pairs:
        ds = set(p["domains"])
        for d in p["domains"]:
            place[d] += 1                       # VNF-placement load
        for d in ds:
            sel[d] += 1                          # arrivals touching d
            if p["admit"]:
                sel_adm[d] += 1
    total_place = sum(place.values()) or 1
    shares = {d: place[d] / total_place for d in range(num_domains)}
    # concentration = normalized entropy of the placement distribution (1=uniform, 0=one domain)
    ps = np.array([shares.get(d, 0.0) for d in range(num_domains)])
    nz = ps[ps > 0]
    H = float(-(nz * np.log(nz)).sum() / np.log(num_domains)) if len(nz) > 1 else 0.0
    top = max(range(num_domains), key=lambda d: shares.get(d, 0.0))
    # saturation on the top domain: admit rate among arrivals touching `top`, first vs second half
    touch_top = [p for p in pairs if top in set(p["domains"])]
    half = len(touch_top) // 2
    fh = touch_top[:half]; sh = touch_top[half:]
    ar = lambda xs: (sum(x["admit"] for x in xs) / len(xs)) if xs else float("nan")

    logger.info("--- %s  (n=%d, overall admit=%.1f%%) ---", label, n,
                100 * sum(p["admit"] for p in pairs) / max(1, n))
    logger.info("  norm-entropy(placement)=%.3f  top-domain=%d share=%.2f", H, top, shares[top])
    logger.info("  per-domain  share | cond-admit(selected)")
    for d in range(num_domains):
        ca = (sel_adm[d] / sel[d]) if sel[d] else float("nan")
        logger.info("    d%d  place=%3d share=%.2f  sel=%3d  cond-admit=%s",
                    d, place[d], shares.get(d, 0.0), sel[d],
                    "n/a " if sel[d] == 0 else f"{100*ca:5.1f}%")
    logger.info("  SATURATION top-domain d%d: admit 1st-half=%s -> 2nd-half=%s (falling=hotspot)",
                top, f"{100*ar(fh):.1f}%" if fh else "n/a",
                f"{100*ar(sh):.1f}%" if sh else "n/a")
    return {"n": n, "norm_entropy": H, "top_domain": top, "shares": shares,
            "place": dict(place), "sel": dict(sel), "sel_admit": dict(sel_adm),
            "sat_top_first": ar(fh), "sat_top_second": ar(sh),
            "overall_admit": sum(p["admit"] for p in pairs) / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="C+_T-_B-")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--arrivals", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bc-scenarios", type=int, default=2000)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--out", default="results/wp7")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"load_concentration_{args.family}.json"
    state = {"family": args.family, "seeds": args.seeds, "rounds": args.rounds,
             "arrivals": args.arrivals, "mu5": MU5, "random_foc_ref": 30.5, "cells": {}}
    if ckpt.exists():
        try:
            state = json.load(open(ckpt)); state.setdefault("cells", {})
            logger.info("resumed: %d seed-cells done", len(state["cells"]))
        except Exception as e:  # noqa: BLE001
            logger.warning("checkpoint unreadable (%s), fresh", e)

    def save():
        json.dump(state, open(ckpt, "w"), indent=2)

    for seed in args.seeds:
        if str(seed) in state["cells"]:
            logger.info("seed %d cached, skip", seed); continue
        t0 = time.time()
        logger.info("### seed %d: BC-warm-start + train RL-alone (gate-identical) ###", seed)
        bc = build_bc_actors(args.family, seed, args.bc_scenarios, args.bc_epochs)
        curve, coord = W.train_arm(
            "RL-alone", args.family, seed, args.rounds, args.arrivals, args.lr,
            0.0, 0.0, None, None, mock=True, actors=bc,
            mdo_cfg=MDOConfig(n_part=3, **MU5), eval_with_train_builder=False,
            return_coord=True)
        num_domains = coord.domain_actors and max(coord.domain_actors) + 1
        num_domains = int(num_domains) if num_domains else 5

        learned, l_adm, l_tot = instrumented_eval(coord, args.family, seed, args.arrivals,
                                                   "deterministic")
        rand, r_adm, r_tot = instrumented_eval(coord, args.family, seed, args.arrivals, "random")

        logger.info("=== seed %d  (train final eval_FoC=%.1f%%) ===",
                    seed, 100 * curve[-1]["eval_foc"])
        l_sum = summarize(learned, num_domains, f"LEARNED seed{seed}")
        r_sum = summarize(rand, num_domains, f"RANDOM  seed{seed}")

        state["cells"][str(seed)] = {
            "num_domains": num_domains,
            "train_final_foc": 100 * curve[-1]["eval_foc"],
            "train_final_entropy": curve[-1]["mdo_entropy"],
            "learned": {"pairs": learned, "admit": l_adm, "total": l_tot, "summary": l_sum},
            "random": {"pairs": rand, "admit": r_adm, "total": r_tot, "summary": r_sum},
        }
        save()
        logger.info("[seed%d] saved (%.0fs). learned normH=%.2f random normH=%.2f",
                    seed, time.time() - t0, l_sum["norm_entropy"], r_sum["norm_entropy"])

    # cross-seed verdict against the pre-registered rule
    logger.info("\n" + "=" * 70)
    logger.info("LOAD-CONCENTRATION VERDICT  family=%s", args.family)
    logger.info("=" * 70)
    for seed in args.seeds:
        c = state["cells"].get(str(seed))
        if not c:
            continue
        ls = c["learned"]["summary"]; rs = c["random"]["summary"]
        conc = ls["norm_entropy"] < rs["norm_entropy"] - 0.05   # learned MORE concentrated
        sat = (not np.isnan(ls["sat_top_first"]) and not np.isnan(ls["sat_top_second"])
               and ls["sat_top_second"] < ls["sat_top_first"] - 0.05)
        logger.info("  seed%d: learned normH=%.2f vs random normH=%.2f  concentrated=%s  "
                    "top-domain admit %.0f%%->%.0f%% saturating=%s",
                    seed, ls["norm_entropy"], rs["norm_entropy"], conc,
                    100 * ls["sat_top_first"], 100 * ls["sat_top_second"], sat)
        if conc and sat:
            logger.info("    -> Mechanism A (hotspot): concentration + decaying admits.")
        elif not conc:
            logger.info("    -> Mechanism B (uninformative obs): no extra concentration vs random.")
        else:
            logger.info("    -> concentration WITHOUT saturation: check conditional-admit flatness (B-leaning).")
    logger.info("Saved: %s", ckpt)
    Path("runs").mkdir(exist_ok=True); Path("runs/LOAD_CONC_DONE").touch()


if __name__ == "__main__":
    main()
