#!/usr/bin/env python
"""Probe: can the KL term move the autoreg policy toward its prior AT ALL?

§V follow-up (2026-07-24), ported to §Y/§Z (2026-08-06). RL-poprior trained with
beta 1->0.2 toward a GOOD partial-obs prior ended at m~-agreement 0.03-0.05 and
train-time KL ~11 nats (anti-aligned; uniform would be ~5). Two rival
explanations:
  (a) the autoreg KL path (build_prior_logits/analytical_kl over conditional
      logits) does not produce a usable gradient -> KL mechanism BROKEN;
  (b) the PPO/reward gradient overwhelms the KL pull -> reward/credit problem.

§Y.9 split (a) further, and this probe now reads that split. `prior_fire_rate`
and `train_mtilde_agreement` separate "the term never fired" from "it fired and
did not align on the states the update saw" from "it aligned on train and did
not transfer to the eval stream". The old verdict line read only the last of
those and so conflated all three.

§Z.5 corrects the readout itself. `train_mtilde_agreement` is teacher-forced
(`evaluate_actions` advances its running counts on the actions handed to it)
while the eval-side `mtilde_agreement` is a free-running decode, so KLS6's two
numbers were never the same measurement and their difference was not transfer.
`train_mtilde_agreement_fr` adds the free-running decode on the train states,
which makes exposure bias (tf minus fr) and transfer (fr minus eval) separable.

§Z.6 fixes the readout twice more. KLS7 showed the scored quantity was one
round of a process that never converges and is bimodal (175 of 200 rounds
below 0.05, 24 above 0.95, one in between), so the gate now scores the mean
over the final `--tail` rounds and reports its spread. It also showed the
ratio has almost no support, because m~ and the deterministic decode both
collapse onto one domain, so the scored agreement is the OFF-MODAL one and a
run whose off-modal support is too small is void rather than failed. Finally
the fixed-stream arm was flat in acceptance across all 200 rounds, so a
learning-signal precondition now decides whether an arm may be scored at all.

§Z.1 ports this off the deleted pre-§Y families onto the §Y surface: one
hierarchical-tree instance, a calibrated load level, acceptance rather than
fraction-of-ceiling. It also equalises the prior-loss reduction, so a beta here
is NOT comparable to a beta in KLS1-KLS5.

LLM-free (partial-obs prior via CUSTOM_PLAN_BUILDER). Writes
data/probe_kl_sanity_<TAG>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Behind __main__ only: fired on import it replaces ANY importer via os.execv,
# which under pytest ends the session with no traceback and rc 0. Same reasoning
# as grid_runner.py:61.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import wp7_runner as W  # noqa: E402
import grid_runner as G  # noqa: E402
from partial_obs_prior import partial_obs_builder  # noqa: E402
from orion.provenance import git_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("probe_kl")

# §Z.6 preconditions, registered before KLS8 fires.
# An arm whose acceptance does not move cannot tell a failed prior term from a
# stalled learner, and an agreement ratio computed on a handful of off-modal
# slots is not a measurement. Either condition voids the arm rather than
# failing it, because both are properties of the substrate and not results.
LEARNING_SIGNAL_MIN = 0.05
OFF_MODAL_SLOTS_MIN = 100
# The Z.6 smoke reads policy_modal_frac 1.0 on every round: the deterministic
# decode puts all 857 slots on one domain. Against a constant policy the
# off-modal number is degenerate too, and in the same direction -- the smoke
# scores off-modal agreement 1.0 on nine slots, purely because those nine
# happen to name the domain the policy always plays. So collapse voids the
# arm ahead of the support check rather than after it.
POLICY_COLLAPSE_MAX = 0.95


def _tail(curve, key, tail, sub_key=None):
    """Mean and sd of a curve field over the last `tail` rounds.

    §Z.6 -- the single-round readout KLS6 and KLS7 were scored on is a draw
    from an oscillation, not a converged value. Returns (mean, sd, n).
    """
    vals = []
    for c in curve[-tail:]:
        v = c.get(key)
        if sub_key is not None:
            v = (v or {}).get(sub_key)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, sd, len(vals)


def run(scenario, level, instance, seed, rounds, arrivals, lr, beta,
        fixed_stream=False, tail=20):
    G._wire(scenario, level, instance)
    W.CUSTOM_PLAN_BUILDER = partial_obs_builder
    W.RC_FIXED_TRAIN_STREAM = fixed_stream
    ck = Path(f"results/wp7/ckpt_grid/klsanity_{scenario}_{seed}_b{beta}.pt")
    curve, coord = W.train_approach(
        "RL-alone", level, seed, rounds, arrivals, lr, beta, beta,
        None, None, mock=True, actors=None,
        entropy_schedule=(0.03, 0.01), eval_with_train_builder=True,
        ckpt_path=str(ck), use_mb=False, return_coord=True)
    last = curve[-1] if curve else {}
    sub = G._substrate_fn(instance)(seed)
    delays = W.build_delays(sub)
    acc, adm, tot, agree = W.eval_acceptance(
        coord, level, seed, arrivals, delays, plan_builder=partial_obs_builder,
        mode="deterministic")
    tf = last.get("train_mtilde_agreement")
    fr = last.get("train_mtilde_agreement_fr")
    # §Z.6 — the SCORED quantities. Tail means, off-modal, with the baselines
    # a pooled ratio has to beat and the precondition that the arm learned at
    # all. The single-round fields below are kept for continuity with KLS6/7
    # and are explicitly not what the gate reads.
    tf_m, tf_sd, _ = _tail(curve, "train_mtilde_agreement", tail)
    fr_m, fr_sd, _ = _tail(curve, "train_mtilde_agreement_fr", tail)
    ev_m, ev_sd, _ = _tail(curve, "mtilde_agreement", tail)
    tf_off, _, _ = _tail(curve, "train_support_tf", tail, "off_modal_agreement")
    fr_off, _, _ = _tail(curve, "train_support_fr", tail, "off_modal_agreement")
    ev_off, _, _ = _tail(curve, "eval_support", tail, "off_modal_agreement")
    off_slots, _, _ = _tail(curve, "eval_support", tail, "off_modal_slots")
    tgt_modal, _, _ = _tail(curve, "eval_support", tail, "target_modal_frac")
    pol_modal, _, _ = _tail(curve, "eval_support", tail, "policy_modal_frac")
    exact, _, _ = _tail(curve, "eval_support", tail, "exact_arrival_frac")
    acc_late, _, _ = _tail(curve, "eval_acceptance", tail)
    acc_early, _, _ = _tail(curve[:tail], "eval_acceptance", tail)
    learning = ((acc_late - acc_early)
                if (acc_late is not None and acc_early is not None) else None)
    scored = {"tail_rounds": tail,
              "tf_tail": tf_m, "tf_tail_sd": tf_sd,
              "fr_tail": fr_m, "fr_tail_sd": fr_sd,
              "eval_tail": ev_m, "eval_tail_sd": ev_sd,
              "tf_off_tail": tf_off, "fr_off_tail": fr_off,
              "eval_off_tail": ev_off, "eval_off_slots_tail": off_slots,
              "target_modal_tail": tgt_modal, "policy_modal_tail": pol_modal,
              "eval_exact_arrival_tail": exact,
              "acceptance_early": acc_early, "acceptance_late": acc_late,
              "learning_signal": learning}
    return {"beta": beta, "rounds": rounds, "scored": scored,
            "kl_final": last.get("kl_mean"), "ent_final": last.get("mdo_entropy"),
            "train_admit_last": last.get("train_admit"),
            "kl_frame_skips": last.get("kl_frame_skips"),
            # §Y.9 fields — the whole point of KLS6.
            "prior_fire_rate": last.get("prior_fire_rate"),
            "train_mtilde_agreement": tf,
            # §Z.5 — free-running on the train states, the metric
            # KLS6 lacked.
            "train_mtilde_agreement_fr": fr,
            "exposure_bias_gap": (tf - fr) if (tf is not None and fr is not None)
                                 else None,
            "eval_acceptance": round(acc, 4), "eval_admit": adm,
            "eval_total": tot, "mtilde_agreement": agree,
            # §Z.5 — KLS6 kept only final values, so the fall in
            # teacher-forced agreement from the smoke could not be told from a
            # difference in arrivals per round. The curve makes it a measurement.
            "curve": [{kk: c.get(kk) for kk in
                       ("round", "prior_fire_rate", "train_mtilde_agreement",
                        "train_mtilde_agreement_fr", "mtilde_agreement",
                        "kl_mean", "mdo_entropy", "eval_acceptance",
                        "train_support_tf", "train_support_fr",
                        "eval_support")}
                      for c in curve]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="conventional")
    ap.add_argument("--level", default=None, help="default: TRAINING_LEVEL")
    ap.add_argument("--instance", type=int, default=None,
                    help="default: first TRAIN_INSTANCES entry (no transfer confound)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--arrivals", type=int, default=None, help="default: NUM_ARRIVALS")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--betas", type=float, nargs="+", default=[25.0])
    ap.add_argument("--fixed-stream", action="store_true",
                    help="repeat the SAME train stream every round (overfit test: "
                         "if KL cannot align even on repeated identical targets, "
                         "the failure is structural, not sample-efficiency)")
    ap.add_argument("--adv-mode", default="td0",
                    choices=["stream_gae", "td0", "bandit"],
                    help="§W.1 advantage construction; td0 is the grid's setting")
    ap.add_argument("--prior-loss", default="distill",
                    choices=["sampled_kl", "distill"],
                    help="§X.4 prior-coupling loss (KLS7 gates distill under §Z.5)")
    ap.add_argument("--tail", type=int, default=20,
                    help="§Z.6: rounds averaged for the scored quantities")
    ap.add_argument("--tag", default="KLS8")
    ap.add_argument("--no-prereg", action="store_true",
                    help="run without the pre-registration, which is not distributed with "
                         "this repository. Applies only when the document is absent; the "
                         "result JSON records prereg.status=\"skipped\".")
    args = ap.parse_args()
    level = args.level or G.TRAINING_LEVEL
    instance = args.instance if args.instance is not None else G.TRAIN_INSTANCES[0]
    arrivals = args.arrivals if args.arrivals is not None else G.NUM_ARRIVALS
    prov = git_provenance(serving=None, tag=args.tag,
                          prereg="docs/PREREG_AMENDMENT_2026-08-06_Z.md",
                          allow_absent_prereg=args.no_prereg)
    log.info("provenance commit=%s dirty=%s", prov["git_commit"][:8], prov["git_dirty"])
    W.ADV_MODE = args.adv_mode
    W.PRIOR_LOSS = args.prior_loss
    log.info("ADV_MODE=%s PRIOR_LOSS=%s lambda_viol=%s level=%s instance=%s "
             "arrivals=%d rounds=%d lr=%s fixed_stream=%s",
             W.ADV_MODE, W.PRIOR_LOSS, W.REWARD_LAMBDA_VIOL, level, instance,
             arrivals, args.rounds, args.lr, args.fixed_stream)
    res = {"provenance": prov, "scenario": args.scenario, "level": level,
           "instance": instance, "seed": args.seed, "adv_mode": args.adv_mode,
           "prior_loss": args.prior_loss, "lr": args.lr,
           "fixed_stream": args.fixed_stream, "arrivals": arrivals,
           "reduction": "sum",  # §Z.1 — betas here are NOT KLS1-5 betas
           "approaches": []}
    for beta in args.betas:
        log.info("### beta=%s", beta)
        r = run(args.scenario, level, instance, args.seed, args.rounds,
                arrivals, args.lr, beta, args.fixed_stream, args.tail)
        res["approaches"].append(r)
        log.info("  %s", r)
        Path("data").mkdir(exist_ok=True)
        Path(f"data/probe_kl_sanity_{args.tag}.json").write_text(
            json.dumps(res, indent=2, default=str))
    print("\nVERDICT (§Z.6 decision rule; scored on tail means, off-modal):")
    for r in res["approaches"]:
        sc = r["scored"]
        fire = r["prior_fire_rate"]
        learn = sc["learning_signal"]
        slots = sc["eval_off_slots_tail"]
        tf = sc["tf_off_tail"]
        fr = sc["fr_off_tail"]
        ev = sc["eval_off_tail"]
        if learn is None or learn < LEARNING_SIGNAL_MIN:
            verdict = ("VOID, NOT A GATE SUBSTRATE -- acceptance did not move "
                       "over the run, so an agreement reading here cannot "
                       "separate a failed prior term from a stalled learner")
        elif fire is None or fire < 0.5:
            verdict = "CHANNEL BARELY CONNECTED -- fix m~ eligibility, re-fire"
        elif (sc["policy_modal_tail"] or 0) >= POLICY_COLLAPSE_MAX:
            verdict = ("VOID, POLICY IS CONSTANT -- the deterministic decode "
                       "plays one domain almost everywhere, so agreement "
                       "reports which domain that is and not whether the "
                       "prior term aligned anything")
        elif slots is None or slots < OFF_MODAL_SLOTS_MIN:
            verdict = ("VOID, NO SUPPORT -- m~ is effectively one domain, so "
                       "the agreement ratio is a coincidence of two collapses")
        elif (fr or 0) >= 0.5 and (ev or 0) >= 0.5:
            verdict = "ALIGNS AND TRANSFERS -- distill becomes default"
        elif (fr or 0) >= 0.5:
            verdict = ("FREE-RUNNING ON TRAIN, NOT ON EVAL -- transfer/"
                       "representation result; §Z.3 does not address it")
        elif (tf or 0) >= 0.5:
            verdict = ("TEACHER-FORCED ONLY -- exposure bias dominates; "
                       "§Z.3 onpolicy_distill is the indicated remedy")
        else:
            verdict = ("ALIGNS UNDER NEITHER DECODE -- dosage/optimisation, "
                       "not the prefix; escalate")
        print(f"  beta={r['beta']} tail={sc['tail_rounds']}")
        print(f"    scored (off-modal): tf={tf} fr={fr} eval={ev} "
              f"off_modal_slots={slots}")
        print(f"    pooled  (tail mean+-sd): tf={sc['tf_tail']}+-{sc['tf_tail_sd']} "
              f"fr={sc['fr_tail']}+-{sc['fr_tail_sd']} "
              f"eval={sc['eval_tail']}+-{sc['eval_tail_sd']}")
        print(f"    baselines: target_modal={sc['target_modal_tail']} "
              f"policy_modal={sc['policy_modal_tail']} "
              f"exact_arrivals={sc['eval_exact_arrival_tail']}")
        print(f"    precondition: fire={fire} acceptance "
              f"{sc['acceptance_early']} -> {sc['acceptance_late']} "
              f"(learning_signal={learn})")
        print(f"    => {verdict}")
    print("PROBE_KL_SANITY_DONE")


if __name__ == "__main__":
    main()
