#!/usr/bin/env python
"""Track D — the 300-arrival adaptation run. Entry point for §T (final hash 1abc402d783b).

Arms, exactly as Δ2-T re-decided them (D.3: cache-OFF, mb=None PRIMARY, because §M's
M.1 measured cache-OFF + M^B-live as the worst known configuration on RC by 10-13
admits/100 -- and it is the one R45's Full-ORION ran):

  1. Full-ORION   trained MDO + LLM plans, cache-OFF, mb=None.   LLM path  -> n=3
  2. RL-alone     greedy plans, beta=0, same trained MDO/actors. deterministic -> n=1
  3. follow_prior cache-OFF, mb=None, no RL. Clean plan quality.  LLM path  -> n=3
  4. Plain-ColocFB over the same 300-arrival streams.             deterministic -> n=1

Firing order is §T's: **trained arms last**. Cheap deterministic references first, so a
mid-run failure still leaves the references banked.

n=3 / median+range on LLM-path arms is EXPERIMENT_PROTOCOL Amendment 9: the serving path
is not run-to-run deterministic at temp 0 (0-5 admits/100 on RC cells). Comparisons under
~10 points are not signal. Repeats vary NOTHING but the serving path -- torch/np seeds are
fixed per cell inside train_arm -- so rep spread isolates exactly that.

RESUMABLE: every cell writes data/d_cells/<arm>_<seed>_rep<r>.json the moment it finishes
and is skipped if that file exists. A 26.7 h run must not lose 20 h to one bad cell; kill
and re-launch is always safe.

Two reference streams are in play and they are NOT interchangeable:
  * arm 3 / Plain-q  -- the q_pilot stream at `seed` (what M.1-none 17/24/12 measured,
    and what the calibration cell ran). This is arm 3's comparator.
  * arms 1-2 / Plain-eval -- wp7's held-out eval stream at `seed+777`, which is where
    FoC-vs-ceiling lives. The in-job follow_prior number reported inside each trained
    cell is same-stream selector isolation on THIS stream (§R Δ2-R), not arm 3.
Both Plain references are computed and labelled; do not cross-compare them.
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

import q_pilot_runner as Q  # noqa: E402
import r_local_runner as R  # noqa: E402
import rc_train_runner as T  # noqa: E402
import wp7_runner as W  # noqa: E402
import five_arm_runner as F  # noqa: E402
import rc_family_validity as V  # noqa: E402
from orion.provenance import git_provenance, serving_provenance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("track_d")

PREREG_T = "docs/PREREG_T_2026-07-16.md"
PREREG_T_SHA = "1abc402d783b083670a4e77b6579e383e78670c3de85e2805151607bcaa2fd51"
CELLS = Path("data/d_cells")

# 100-arrival comparators, for the readout only. Arm 3's is M.1-none (Δ2-T's
# "comparator correction"): R.2-prime's R.2 cells ran M^B LIVE and are the wrong anchor.
M1_NONE_100 = {42: 17, 43: 24, 44: 12}


def _cell_path(arm, seed, rep):
    return CELLS / f"{arm}_{seed}_rep{rep}.json"


def _done(arm, seed, rep):
    p = _cell_path(arm, seed, rep)
    if p.exists():
        logger.info("SKIP %s seed=%d rep=%d (already banked: %s)", arm, seed, rep, p)
        return True
    return False


def _bank(arm, seed, rep, payload, prov):
    CELLS.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, arm=arm, seed=seed, rep=rep, provenance=prov)
    _cell_path(arm, seed, rep).write_text(json.dumps(payload, indent=2, default=str))
    logger.info("BANKED %s", _cell_path(arm, seed, rep))


def plain_reference(seed, arrivals, prov):
    """Arm 4. Deterministic, no LLM, no RL -> n=1.

    Uses rc_family_validity._evaluate, which is how the frozen Plain reference (40.4)
    was computed in the first place: each arrival evaluated independently against the
    empty substrate (a validity draw, not a stateful episode). Reusing it keeps the
    300-arrival number comparable in kind to the 100-arrival one.
    """
    if _done("Plain", seed, 0):
        return
    sub = T.rc_substrate_fn(seed)
    V.ARRIVALS = arrivals
    out = {}
    for label, astream in (("q", seed), ("eval", seed + 777)):
        t0 = time.time()
        total, ceiling, plain = V._evaluate(sub, arrival_seed=astream)
        out[label] = {"stream_seed": astream, "total": total, "ceiling": ceiling,
                      "plain_admits": plain,
                      "plain_foc": 100.0 * plain / ceiling if ceiling else float("nan"),
                      "wall_s": round(time.time() - t0, 1)}
        logger.info("Plain[%s] seed=%d stream=%d: total=%d ceiling=%d plain=%d FoC=%.1f%%",
                    label, seed, astream, total, ceiling, plain, out[label]["plain_foc"])
    _bank("Plain", seed, 0, out, prov)


def follow_prior_cell(seed, arrivals, agent_b, kb, rep, prov):
    """Arm 3: follow_prior, cache-OFF, mb=None, on the q_pilot stream at `seed`."""
    if _done("follow_prior", seed, rep):
        return
    Q.ARRIVALS_PER_INSTANCE = arrivals
    sub = Q.generate_rc_instance(seed=Q.RC_GEN_SEED,
                                 inter_domain_bw_override=R.FROZEN_RC[seed]["bw"])
    t0 = time.time()
    m = Q.run_q_cell(f"D-follow_prior-rep{rep}", agent_b, kb, None, None, sub, seed, False)
    wall = time.time() - t0
    logger.info("follow_prior seed=%d rep=%d: %d/%d admits (%.1f min)",
                seed, rep, m["admitted"], m["total"], wall / 60)
    _bank("follow_prior", seed, rep, {
        "admitted": m["admitted"], "total": m["total"],
        "admit_pct": 100.0 * m["admitted"] / max(1, m["total"]),
        "reasons": dict(m.get("reasons", {})), "wall_s": round(wall, 1),
        "cache_on": False, "mb": None, "comparator_100": M1_NONE_100.get(seed),
    }, prov)


def trained_cell(arm, seed, rounds, arrivals, args, agent_b, kb, rep, prov):
    """Arms 1-2. Full-ORION is the LLM path (n=3); RL-alone is deterministic (n=1)."""
    if _done(arm, seed, rep):
        return
    actors = None if args.actors == "greedy" else T.build_rc_bc_actors(
        seed, args.bc_scenarios, args.bc_epochs)
    ckpt_dir = Path("results/wp7/ckpt_D")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if arm == "RL-alone":
        # No LLM to memoize -> varied per-round streams (its validated protocol).
        W.RC_FIXED_TRAIN_STREAM = False
        curve = W.train_arm("RL-alone", T.RC_FAM, seed, rounds, arrivals, args.lr,
                            0.0, 0.0, None, None, mock=True, actors=actors,
                            entropy_schedule=(0.03, 0.01),
                            ckpt_path=str(ckpt_dir / f"RL-alone_{seed}.pt"))
    else:
        # Fixed stream + content plan-memo => ~arrivals LLM calls total, not
        # arrivals*rounds. use_mb=False is D.3.
        W.RC_FIXED_TRAIN_STREAM = True
        curve = W.train_arm("LLM+RL-full", T.RC_FAM, seed, rounds, arrivals, args.lr,
                            1.0, 0.0, agent_b, kb, mock=args.mock, actors=actors,
                            entropy_schedule=(0.03, 0.01), eval_with_train_builder=True,
                            ckpt_path=str(ckpt_dir / f"Full-ORION_{seed}_rep{rep}.pt"),
                            use_mb=False)
    wall = time.time() - t0
    cell = T._cell(curve)
    logger.info("%s seed=%d rep=%d: FoC=%.1f%% follow_prior=%.1f%% delta=%+.1fpp (%.1f min)",
                arm, seed, rep, cell["foc_trained"], cell["foc_follow_prior"],
                cell["selector_delta_pp"], wall / 60)
    cell["wall_s"] = round(wall, 1)
    cell["use_mb"] = False
    cell["cache_on"] = bool(W.RC_USE_PLAN_CACHE)
    _bank(arm, seed, rep, cell, prov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--reps", type=int, default=3, help="Amendment 9: n=3 on LLM-path arms")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--arrivals", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--actors", choices=["bc", "greedy"], default="bc")
    ap.add_argument("--bc-scenarios", type=int, default=2000)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--arms", nargs="+", default=["Plain", "follow_prior", "RL-alone", "Full-ORION"],
                    choices=["Plain", "follow_prior", "RL-alone", "Full-ORION"])
    ap.add_argument("--mock", action="store_true", help="FFD stand-in for the LLM (no server)")
    ap.add_argument("--tag", default="D")
    args = ap.parse_args()

    prov = git_provenance(serving=serving_provenance(args.port), tag=args.tag,
                          prereg=PREREG_T, cited_prereg_sha256=PREREG_T_SHA)
    logger.info("provenance: commit=%s dirty=%s prereg_T=%s serving=%s",
                prov["git_commit"][:8], prov["git_dirty"],
                prov["prereg"]["sha256"][:12], prov["serving"])

    W.RC_SUBSTRATE_FN = T.rc_substrate_fn
    W.RC_SLICE_FACTORY = T.rc_slice_factory
    W.RC_USE_PLAN_CACHE = False          # D.3: cache-OFF

    agent_b, kb = None, None
    if not args.mock and ({"follow_prior", "Full-ORION"} & set(args.arms)):
        agent_b = R._build_local_agent(args.port)
        kb = R._load_kb()

    logger.info("TRACK D: seeds=%s reps=%s rounds=%d arrivals=%d arms=%s",
                args.seeds, args.reps, args.rounds, args.arrivals, args.arms)

    # §T firing order: deterministic references first, TRAINED ARMS LAST.
    t_start = time.time()
    if "Plain" in args.arms:
        for seed in args.seeds:
            plain_reference(seed, args.arrivals, prov)
    if "follow_prior" in args.arms:
        for seed in args.seeds:
            for rep in range(1, args.reps + 1):
                follow_prior_cell(seed, args.arrivals, agent_b, kb, rep, prov)
    if "RL-alone" in args.arms:
        for seed in args.seeds:
            trained_cell("RL-alone", seed, args.rounds, args.arrivals, args, None, None, 1, prov)
    if "Full-ORION" in args.arms:
        for seed in args.seeds:
            for rep in range(1, args.reps + 1):
                trained_cell("Full-ORION", seed, args.rounds, args.arrivals, args,
                             agent_b, kb, rep, prov)

    logger.info("TRACK D DONE in %.1f h", (time.time() - t_start) / 3600)
    _readout(args)


def _med_range(vals):
    if not vals:
        return "n/a"
    v = sorted(vals)
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    return f"{med:.1f} [{v[0]:.1f}-{v[-1]:.1f}]" if len(v) > 1 else f"{v[0]:.1f} (n=1)"


def _load(arm, seed):
    return [json.loads(p.read_text()) for p in sorted(CELLS.glob(f"{arm}_{seed}_rep*.json"))]


def _readout(args):
    print("\n" + "=" * 78)
    print("TRACK D — 300-arrival RC. LLM-path arms: median [min-max] over n=%d (Amendment 9)." % args.reps)
    print("Comparisons under ~10 points are NOT signal.")
    print("=" * 78)
    for seed in args.seeds:
        pl = _load("Plain", seed)
        print(f"\nseed {seed} (bw={R.FROZEN_RC[seed]['bw']}):")
        if pl:
            for label in ("q", "eval"):
                r = pl[0].get(label, {})
                if r:
                    print(f"  Plain-ColocFB [{label:4s} stream {r['stream_seed']}] : "
                          f"{r['plain_admits']}/{r['total']} admits, ceiling {r['ceiling']}, "
                          f"FoC {r['plain_foc']:.1f}%")
        fp = _load("follow_prior", seed)
        if fp:
            print(f"  arm3 follow_prior (q stream, mb=None) : "
                  f"{_med_range([c['admit_pct'] for c in fp])}%  "
                  f"[100-arrival comparator M.1-none = {M1_NONE_100.get(seed)}/100]")
        for arm in ("RL-alone", "Full-ORION"):
            cs = _load(arm, seed)
            if cs:
                print(f"  {arm:11s} FoC (eval stream)        : "
                      f"{_med_range([c['foc_trained'] for c in cs])}%   "
                      f"in-job follow_prior {_med_range([c['foc_follow_prior'] for c in cs])}%   "
                      f"selector_delta {_med_range([c['selector_delta_pp'] for c in cs])}pp")
    print("\n  NOTE: arm3 follow_prior runs the q stream (seed); the in-job follow_prior above")
    print("  runs wp7's held-out eval stream (seed+777). Same protocol, DIFFERENT streams —")
    print("  do not cross-compare them. The in-job one is §R Δ2-R selector isolation.")


if __name__ == "__main__":
    main()
