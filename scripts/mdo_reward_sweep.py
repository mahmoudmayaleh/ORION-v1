#!/usr/bin/env python3
"""MDO reward-rebalance sweep (Layer-1 fix from the plateau check).

The plateau check showed the ORION reward's cost/trial/quality terms (alpha=xi=eta=0.1)
drag the learned MDO to 17.2% -- 14 pts BELOW masked-random (31%) -- and stripping them
(admission-pure) recovers to ~31%. This sweep answers the follow-ups that pre-register the
reward config for every downstream axis:

  1. Is FoC monotone in the cost scale c (alpha=xi=eta=c)?  -> confirms the mechanism.
  2. Does raising the admission weight mu (cost fixed) recover instead of zeroing cost?
  3. WHICH term dominates -- alpha (cost) / xi (trial) / eta (quality)?  (isolate each.)
  4. Does ANY mu:alpha:xi:eta beat masked-random (~31%)?  If none, Layer-2 (action space)
     is the hard ceiling and reward tuning alone tops out at the random floor.

Same harness as mdo_plateau_check (frozen GreedyDomainActor, RL-alone beta=0,
contextual-bandit advantage) -- ONLY the MDOConfig reward weights vary. 3 seeds (FoC is a
noisy floor; single-seed last-10 means bounce +/- several pts). Incremental checkpoint:
re-running resumes, skipping finished (config, seed) pairs.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
from mdo_plateau_check import train_mdo, baseline_foc, build_delays, FAM
from orion.mdo.coordinator import MDOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sweep")

# (label, mu, alpha, xi, eta)
CONFIGS = [
    ("cost0.00_pure",   1.0, 0.00, 0.00, 0.00),   # = admission-pure (upper anchor)
    ("cost0.02",        1.0, 0.02, 0.02, 0.02),
    ("cost0.05",        1.0, 0.05, 0.05, 0.05),
    ("cost0.10_ORION",  1.0, 0.10, 0.10, 0.10),    # = baseline ORION (lower anchor, 17.2%)
    ("cost0.20",        1.0, 0.20, 0.20, 0.20),
    ("mu2_cost0.10",    2.0, 0.10, 0.10, 0.10),     # raise admission weight instead of zeroing cost
    ("mu5_cost0.10",    5.0, 0.10, 0.10, 0.10),
    ("alpha_only0.10",  1.0, 0.10, 0.00, 0.00),     # isolate cost term
    ("xi_only0.10",     1.0, 0.00, 0.10, 0.00),     # isolate trial term
    ("eta_only0.10",    1.0, 0.00, 0.00, 0.10),     # isolate quality term
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="C+_T-_B-")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--arrivals", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--out", default="results/wp7")
    args = p.parse_args()

    sub0 = R.generate_family_instance(FAM[args.family], seed=0)
    delays = build_delays(sub0)
    _, ceiling = R.compute_ceiling(sub0, args.seeds[0] + 777)
    logger.info("MDO reward sweep: family=%s seeds=%s rounds=%d ceiling=%d configs=%d",
                args.family, args.seeds, args.rounds, ceiling, len(CONFIGS))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"mdo_reward_sweep_{args.family}.json"
    state = {"family": args.family, "seeds": args.seeds, "rounds": args.rounds,
             "ceiling": ceiling, "cells": {}, "random_by_seed": {}}
    if ckpt.exists():
        try:
            state = json.load(open(ckpt))
            state.setdefault("cells", {}); state.setdefault("random_by_seed", {})
            logger.info("resumed: %d cells done", len(state["cells"]))
        except Exception as e:
            logger.warning("checkpoint unreadable (%s), starting fresh", e)

    def save():
        json.dump(state, open(ckpt, "w"), indent=2)

    # Layer-2 reference: masked-random per seed.
    for seed in args.seeds:
        if str(seed) in state["random_by_seed"]:
            continue
        rf = 100 * baseline_foc(args.family, seed, args.arrivals, ceiling, delays, "random", None)
        state["random_by_seed"][str(seed)] = float(rf)
        logger.info("random(seed=%d) = %.1f%%", seed, rf)
        save()

    total = len(CONFIGS) * len(args.seeds)
    done0 = len(state["cells"])
    for label, mu, alpha, xi, eta in CONFIGS:
        for seed in args.seeds:
            key = f"{label}|{seed}"
            if key in state["cells"]:
                continue
            t0 = time.time()
            _, _, curve = train_mdo(args.family, seed, args.rounds, args.arrivals, args.lr,
                                    MDOConfig(n_part=3, mu=mu, alpha=alpha, xi=xi, eta=eta),
                                    ceiling, delays, label)
            foc = float(np.mean([c["eval_foc"] for c in curve[-10:]]) * 100)
            ent = float(np.mean([c["mdo_entropy"] for c in curve[-10:]]))
            state["cells"][key] = {"label": label, "seed": seed, "mu": mu, "alpha": alpha,
                                   "xi": xi, "eta": eta, "foc": foc, "entropy": ent}
            save()
            logger.info("[%d/%d] %-16s seed=%d FoC=%.1f%% ent=%.3f (%.0fs)",
                        len(state["cells"]) - done0, total - done0, label, seed, foc, ent,
                        time.time() - t0)

    # Aggregate mean +/- std across seeds.
    logger.info("\n" + "=" * 68)
    logger.info("MDO REWARD SWEEP  family=%s  (ceiling=%d)", args.family, ceiling)
    rnd_mean = float(np.mean(list(state["random_by_seed"].values())))
    logger.info("masked-random (Layer-2 reference) = %.1f%%   Plain-ColocFB = 85.3%%", rnd_mean)
    logger.info("=" * 68)
    logger.info("%-16s %8s %8s   %s", "config", "FoC", "vs rand", "(mu,a,xi,eta)")
    agg = {}
    for label, mu, alpha, xi, eta in CONFIGS:
        focs = [state["cells"][f"{label}|{s}"]["foc"] for s in args.seeds
                if f"{label}|{s}" in state["cells"]]
        if not focs:
            continue
        m, sd = float(np.mean(focs)), float(np.std(focs))
        agg[label] = {"mean": m, "std": sd, "n": len(focs), "vs_random": m - rnd_mean}
        logger.info("%-16s %5.1f+-%-3.0f %+7.1f   (%.0f,%.2f,%.2f,%.2f)",
                    label, m, sd, m - rnd_mean, mu, alpha, xi, eta)
    state["aggregate"] = agg
    state["random_mean"] = rnd_mean
    save()

    best = max(agg.items(), key=lambda kv: kv[1]["mean"]) if agg else (None, None)
    if best[0]:
        logger.info("-" * 68)
        logger.info("BEST: %s = %.1f%% (%+.1f vs random). %s", best[0], best[1]["mean"],
                    best[1]["vs_random"],
                    "beats random -> shaping helps beyond feasibility"
                    if best[1]["vs_random"] > 1.0 else
                    "<= random -> Layer-2 (action space) is the hard ceiling, reward tops out at floor")
    logger.info("Saved: %s", ckpt)
    Path("runs").mkdir(exist_ok=True); Path("runs/MDO_REWARD_SWEEP_DONE").touch()


if __name__ == "__main__":
    main()
