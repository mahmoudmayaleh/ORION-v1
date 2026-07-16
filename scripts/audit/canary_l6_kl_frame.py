#!/usr/bin/env python3
"""Ladder L6 — behavioral check for §O.4 (KL prior frame) + §O.5 (agreement metric).

L1-L5 all run beta=0, so the KL path is otherwise unexercised by the ladder.
Here: gate family C+_T-_B- (canonical sort permutes [1,3,2,0,4]), train_arm
with a STRONG CONSTANT KL prior (beta=5, no annealing) toward the greedy m~
(mock mode). If the prior is built in the correct (canonical) frame, KL
regularization must drive the policy toward m~: mtilde_agreement (computed in
one frame per §O.5) rises far above the 1/M = 20% uniform baseline. Pre-O
this could not happen: the prior peaked on the wrong domain in 4/5 positions,
so pushing toward it pushed AWAY from m~.

Pass bar — calibrated to the prior itself: build_prior_logits is
temperature-softened (T=1): prior logits are 1.0 on m~ and 0.0 elsewhere, so
over 5 feasible domains the prior's OWN m~ mass is e/(e+4) = 0.405. Perfect
KL convergence therefore yields agreement ~= 0.405, NOT ~1.0 (first-run bar
of 0.6 was miscalibrated against a sharp prior that build_prior_logits
deliberately does not produce). Pass: tail-3 agreement >= 0.35 (prior peak
minus margin; uniform baseline 0.20; pre-O permuted frame would sit ~0.13
since only 1 of 5 canonical positions is a fixed point of [1,3,2,0,4]) AND
kl_frame_skips == 0 AND KLs finite AND mean(last-3 KL) < mean(first-3 KL).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import wp7_runner as W
from orion.mdo.coordinator import MDOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("l6")

FAMILY = "C+_T-_B-"
SEED = 42
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ARRIVALS = 30
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)


def main():
    curve = W.train_arm(
        "LLM+RL-memoff", FAMILY, SEED, ROUNDS, ARRIVALS,
        lr=3e-3, beta_start=5.0, beta_end=5.0, agent_b=None, kb=None,
        mock=True,  # greedy m~ (raw domain IDs) — the frame under test
        actors=None, mdo_cfg=MDOConfig(n_part=3, **MU5),
        eval_with_train_builder=False, return_coord=False,
        entropy_schedule=None, train_trace_path=None,
    )
    agree = [c["mtilde_agreement"] for c in curve if c["mtilde_agreement"] is not None]
    kls = [c["kl_mean"] for c in curve]
    skips = [c.get("kl_frame_skips", 0) for c in curve]
    tail3 = float(np.mean(agree[-3:])) if len(agree) >= 3 else 0.0

    prior_peak_mass = float(np.e / (np.e + 4))  # 0.405 — T=1 soft prior over 5 domains
    kl_decreasing = (len(kls) >= 6 and
                     float(np.mean(kls[-3:])) < float(np.mean(kls[:3])))
    result = {
        "canary": "L6-kl-frame", "family": FAMILY, "seed": SEED,
        "rounds": ROUNDS, "beta": 5.0,
        "mtilde_agreement_curve": agree, "kl_mean_curve": kls,
        "kl_frame_skips": skips,
        "tail3_agreement": tail3,
        "prior_peak_mass": prior_peak_mass,
        "kl_decreasing_first3_to_last3": kl_decreasing,
        "pass": (tail3 >= 0.35 and all(s == 0 for s in skips)
                 and all(np.isfinite(k) for k in kls)
                 and kl_decreasing),
    }
    out = HERE / "out" / "canary_l6_result.json"
    json.dump(result, open(out, "w"), indent=2)
    log.info("agreement curve: %s", [round(a, 2) for a in agree])
    log.info("kl_mean curve:   %s", [round(k, 3) for k in kls])
    log.info("tail-3 agreement=%.2f (prior peak mass %.3f, uniform 0.20, pre-O frame ~0.13) "
             "kl_decreasing=%s skips=%s -> %s",
             tail3, prior_peak_mass, kl_decreasing, sum(skips),
             "PASS" if result["pass"] else "FAIL")
    log.info("saved: %s", out)


if __name__ == "__main__":
    main()
