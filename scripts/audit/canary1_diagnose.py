#!/usr/bin/env python3
"""Fault localization for the Canary-1 failure (instrumentation only).

Reruns Canary 1 (constant-best, no saturation) through the UNTOUCHED gate
trainer path, with wp7_runner.gae_over_arrivals wrapped so that per round we
record — from the exact tensors the update consumes:

  - critic explained variance: 1 - Var(returns - values) / Var(returns)
    (the directive's Phase-3 statistic, computed at arrival granularity)
  - corr(advantage, arrival ordinal): if advantages are dominated by stream
    position rather than action quality, this is strongly negative
  - fraction of ADMITTED arrivals (reward > mu/2) whose broadcast advantage
    is negative: these arrivals' correct actions are pushed DOWN by the update

The wrapper calls the original function and returns its result unchanged —
zero divergence from the audited path.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from orion.mdo.coordinator import MDOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("canary1diag")

FAMILY = "C+_T+_B+"
GOOD_DOMAIN = 2
TINY = 0.05
SEED = 42
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ARRIVALS = 45
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)

_orig_gen = R.generate_family_instance

def _canary_gen(family, seed, **kw):
    sub = _orig_gen(family, seed, **kw)
    for n, d in sub.graph.nodes(data=True):
        if d.get("domain_id") != GOOD_DOMAIN:
            d["cpu_capacity"] = TINY
            d["ram_capacity"] = TINY
            d["cpu_residual"] = min(d["cpu_residual"], TINY)
            d["ram_residual"] = min(d["ram_residual"], TINY)
    return sub

R.generate_family_instance = _canary_gen
W.SERVICE_RATE = 2.0

DIAG = []
_orig_gae = W.gae_over_arrivals


def _instrumented_gae(buffer, trial_values, gamma, lam):
    advantages, returns = _orig_gae(buffer, trial_values, gamma, lam)

    dones = buffer.dones
    T = len(dones)
    arrival_of, a = [0] * T, 0
    for i in range(T):
        arrival_of[i] = a
        if dones[i] >= 0.5:
            a += 1
    term_idx = [i for i in range(T) if dones[i] >= 0.5]

    ar_ret = np.array([float(returns[i]) for i in term_idx])
    ar_val = np.array([float(trial_values[i]) for i in term_idx])
    ar_adv = np.array([float(advantages[i]) for i in term_idx])
    ar_rew = np.array([buffer.rewards[i] for i in term_idx])
    ordinal = np.arange(len(term_idx), dtype=float)

    var_ret = float(np.var(ar_ret))
    ev = 1.0 - float(np.var(ar_ret - ar_val)) / var_ret if var_ret > 1e-12 else float("nan")
    corr_pos = (float(np.corrcoef(ar_adv, ordinal)[0, 1])
                if len(ordinal) > 2 and np.std(ar_adv) > 1e-12 else float("nan"))
    admitted = ar_rew > MU5["mu"] / 2.0
    neg_adv_admitted = (float(np.mean(ar_adv[admitted] < 0.0))
                        if admitted.any() else float("nan"))
    corr_rew = (float(np.corrcoef(ar_adv, ar_rew)[0, 1])
                if len(ar_rew) > 2 and np.std(ar_adv) > 1e-12 and np.std(ar_rew) > 1e-12
                else float("nan"))

    DIAG.append({
        "round": len(DIAG) + 1,
        "n_arrivals": len(term_idx),
        "n_admitted": int(admitted.sum()),
        "explained_variance": ev,
        "corr_adv_vs_position": corr_pos,
        "corr_adv_vs_reward": corr_rew,
        "frac_admitted_with_negative_adv": neg_adv_admitted,
        "returns_head_tail": [float(ar_ret[0]), float(ar_ret[-1])],
        "values_head_tail": [float(ar_val[0]), float(ar_val[-1])],
    })
    return advantages, returns


W.gae_over_arrivals = _instrumented_gae


def main():
    W.train_arm(
        "RL-alone", FAMILY, SEED, ROUNDS, ARRIVALS,
        lr=3e-3, beta_start=0.0, beta_end=0.0, agent_b=None, kb=None, mock=True,
        actors=None, mdo_cfg=MDOConfig(n_part=3, **MU5),
        eval_with_train_builder=False, return_coord=True,
        entropy_schedule=None, train_trace_path=None,
    )
    out = HERE / "out" / "canary1_diagnostics.json"
    json.dump(DIAG, open(out, "w"), indent=2)
    log.info("%-3s %-4s %-8s %-9s %-9s %-9s  ret[0]->ret[-1]  V[0]->V[-1]",
             "rnd", "adm", "EV", "corr_pos", "corr_rew", "negAdv@adm")
    for d in DIAG:
        log.info("%-3d %-4d %-8.3f %-9.3f %-9.3f %-9.3f  %8.1f->%6.1f  %8.1f->%6.1f",
                 d["round"], d["n_admitted"], d["explained_variance"],
                 d["corr_adv_vs_position"], d["corr_adv_vs_reward"],
                 d["frac_admitted_with_negative_adv"],
                 d["returns_head_tail"][0], d["returns_head_tail"][1],
                 d["values_head_tail"][0], d["values_head_tail"][1])
    log.info("saved: %s", out)


if __name__ == "__main__":
    main()
