#!/usr/bin/env python3
"""Run #2 (merged gate) — does the MDO learn useful partitions, and does the LLM prior matter?

Supersedes Probe A (BC-actor plateau) by MERGING it in: the actor hypothesis is now a
FIXED condition (BC-warm-started actors across all arms), not a separate run, so it cannot
confound the prior read-out. Pre-registered amendment: docs/PREREG_AMENDMENT_2026-07-10.md.

Everything below is FIXED across arms (differ ONLY in the stated variable):
  reward     = mu5 admission-dominant  (MDOConfig mu=5, alpha=xi=eta=0.1)  -- restores the
               Remark-4 weight-dominance condition mu >> alpha,gamma the shipped (1,.1,.1,.1)
               violated (see MDO_REWARD_SWEEP_RESULT.md).
  actors     = BC-warm-started DomainActors, FROZEN (shared object across arms per seed).
  n_part     = 3 (unchanged; n_part-down is a follow-up single-variable probe IF this gate
               passes, NOT part of it).
  family     = C+_T-_B-, seeds 42/43/44, byte-identical arrival + eval streams per seed.

Arms:
  (1) RL-alone   : no prior (beta=0, LLM-free greedy m~ for obs/mask only), MDO learns.
                   -> does reward-fix + BC actors alone lift RL off the ~30% random floor?
  (2) LLM+RL     : KL prior toward Agent B's suggestion m~ (beta 1->0), MDO learns. THE SYSTEM.
  (3) Prior-only : NO learning. follow_prior on Agent B's m~ (n_part=1). The control that
                   protects the claim: is the MDO adding anything over executing the prior?

Reference lines (not arms): Plain-ColocFB = 85.3%, masked-random = 30.5%.

PRE-NAMED READ-OUTS (write BEFORE the numbers land):
  (2) > (3) AND (2) > (1)         -> headline contest live & honest: RL learns useful
                                     deviations from the prior AND the prior matters.
  (2) ~= (3)                      -> MDO adds nothing over the prior at this scale; the
                                     bottleneck hunt moves to credit assignment / observation
                                     BEFORE anything full-scale.
  (1) ~= random even w/ BC+mu5    -> the RL problem itself is broken; no prior result is
                                     publishable on top of it.
  (1) lifts substantially         -> the actor hypothesis was the real story; the earlier
                                     "RL can't beat random" framing dies (fine -- it should).
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
import wp7_runner as W
from probe_a_plateau import build_bc_actors
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.profiling import PowerSampler, ProfileCollector, set_collector
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gate")

FAM = {f.short_name: f for f in R.ALL_FAMILIES}
PLAIN_FOC = 85.3
RANDOM_FOC = 30.5
MU5 = dict(mu=5.0, alpha=0.1, xi=0.1, eta=0.1)  # admission-dominant (Remark 4)


def prior_only_eval(family, seed, arrivals, ceiling, delays, plan_builder, actors):
    """FoC of executing Agent B's suggestion with NO learning: follow_prior (n_part=1)
    on the SAME held-out eval stream (seed+777) the trained arms use."""
    sub = R.generate_family_instance(FAM[family], seed=0)
    coord = MDOCoordinator(None, actors, MDOConfig(n_part=1, **MU5))
    rng = np.random.default_rng(seed + 777)
    ap = ArrivalProcess(sub, arrivals, W.ARRIVAL_RATE, W.SERVICE_RATE, rng)
    ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=plan_builder)
    runner.reset()
    ep = runner.run_episode(mdo_mode="follow_prior")
    return (ep.stats.admitted / ceiling if ceiling > 0 else 0.0), ep.stats.admitted, ep.stats.total_arrivals


def git_state():
    """§O Δ3 — provenance: commit hash + dirty flag for the result JSON."""
    # Shared recorder: refuses on untracked code under scripts/ or src/ (the hole the
    # --untracked-files=no form left open), and fails closed if git itself errors --
    # the old form swallowed every exception and returned (None, None), recording an
    # unprovenanced run as merely "unknown" instead of stopping it.
    from orion.provenance import git_provenance
    prov = git_provenance()
    return prov["git_commit"], prov["git_dirty"]


def ev_tail5_of(curve):
    """§O Δ2 — EV(tail-5) from the cell's own §O.8 telemetry."""
    evs = [c.get("ev") for c in curve
           if isinstance(c.get("ev"), (int, float)) and c.get("ev") == c.get("ev")]
    return float(np.mean(evs[-5:])) if len(evs) >= 5 else float("nan")


def stamp_validity(cell, curve):
    """§O Δ2 — learning cell is a VALID EXECUTION only if EV(tail-5) >= 0.5.
    Invalid cells are recorded as invalid-execution, NOT as nulls."""
    ev5 = ev_tail5_of(curve)
    cell["ev_tail5"] = ev5
    cell["valid_execution"] = bool(ev5 == ev5 and ev5 >= 0.5)
    return cell


def plateau_of(curve):
    return float(np.mean([c["eval_foc"] for c in curve[-10:]]) * 100)


def entropy_of(curve):
    return float(np.mean([c["mdo_entropy"] for c in curve[-10:]]))


def _run_arm(arm, args, seed, bc_actors, agent_b, kb, ceiling, delays, ent_sched,
             traces_dir=None, ckpt_dir=None):
    """Execute one arm, return its cell dict (or None if arm unknown). Profiling collector is
    set by the caller, so all episode decisions inside are captured."""
    def _train_trace(a):
        if traces_dir is None:
            return None
        traces_dir.mkdir(parents=True, exist_ok=True)
        return traces_dir / f"{a}_{seed}_train.jsonl"

    def _ckpt(a):  # §O.7 — learning arms checkpoint policy+critic+value-norm
        if ckpt_dir is None:
            return None
        return ckpt_dir / f"{a}_{seed}.pt"

    if arm == "RL-alone":
        curve, coord = W.train_arm("RL-alone", args.family, seed, args.rounds, args.arrivals,
                                   args.lr, 0.0, 0.0, None, None, mock=True, actors=bc_actors,
                                   mdo_cfg=MDOConfig(n_part=3, **MU5), eval_with_train_builder=False,
                                   entropy_schedule=ent_sched, return_coord=True,
                                   train_trace_path=_train_trace("RL-alone"),
                                   ckpt_path=_ckpt("RL-alone"))
        # Mandatory behavioral eval (§N.2): learned (deterministic) vs random reference,
        # byte-identical stream. Feeds criterion (b) + the k-analysis directly.
        trace = {
            "learned": W.instrumented_eval(coord, args.family, seed, args.arrivals, delays,
                                           plan_builder=None, mode="deterministic"),
            "random": W.instrumented_eval(coord, args.family, seed, args.arrivals, delays,
                                          plan_builder=None, mode="random"),
        }
        return stamp_validity({"arm": arm, "seed": seed, "foc": plateau_of(curve),
                               "entropy": entropy_of(curve), "curve": curve,
                               "trace": trace}, curve)
    if arm == "LLM+RL":
        pb = W.make_llm_plan_builder(agent_b, kb, lambda: None)  # M^B off, matches training
        curve, coord = W.train_arm("LLM+RL-memoff", args.family, seed, args.rounds, args.arrivals,
                                   args.lr, args.beta_start, args.beta_end, agent_b, kb, mock=False,
                                   actors=bc_actors, mdo_cfg=MDOConfig(n_part=3, **MU5),
                                   eval_with_train_builder=True, entropy_schedule=ent_sched,
                                   return_coord=True, train_trace_path=_train_trace("LLM+RL"),
                                   ckpt_path=_ckpt("LLM+RL"))
        trace = {
            "learned": W.instrumented_eval(coord, args.family, seed, args.arrivals, delays,
                                           plan_builder=pb, mode="deterministic"),
            "random": W.instrumented_eval(coord, args.family, seed, args.arrivals, delays,
                                          plan_builder=pb, mode="random"),
        }
        return stamp_validity({"arm": arm, "seed": seed, "foc": plateau_of(curve),
                               "entropy": entropy_of(curve), "curve": curve,
                               "trace": trace}, curve)
    if arm == "Prior-only":
        pb = W.make_llm_plan_builder(agent_b, kb, lambda: None)  # M^B off
        foc, adm, tot = prior_only_eval(args.family, seed, args.arrivals, ceiling, delays,
                                        pb, bc_actors)
        return {"arm": arm, "seed": seed, "foc": 100 * foc, "eval_admit": adm,
                "eval_total": tot, "entropy": None}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="C+_T-_B-")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--arrivals", type=int, default=45)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--beta-start", type=float, default=1.0)
    ap.add_argument("--beta-end", type=float, default=0.0)
    ap.add_argument("--bc-scenarios", type=int, default=2000)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--arms", nargs="+", default=["RL-alone", "LLM+RL", "Prior-only"])
    ap.add_argument("--out", default="results/wp7")
    ap.add_argument("--tag", default="", help="output-name suffix; use a NEW tag for the "
                    "amended (h^m + entropy-floor) run so it cannot resume pre-enrichment cells")
    ap.add_argument("--ent-c0", type=float, default=None, help="entropy-schedule c0 (PREREG "
                    "§M.4-Δ5); if set with --ent-floor, learning arms use max(floor, c0*(1-r/(R-1)))")
    ap.add_argument("--ent-floor", type=float, default=0.01)
    args = ap.parse_args()
    ent_sched = (args.ent_c0, args.ent_floor) if args.ent_c0 is not None else None
    if ent_sched:
        logger.info("entropy-floor schedule ACTIVE: c0=%.3f floor=%.3f (learning arms)",
                    ent_sched[0], ent_sched[1])

    # Agent B + K^B (shared) for arms 2 & 3.
    need_llm = any(a in ("LLM+RL", "Prior-only") for a in args.arms)
    agent_b, kb = None, None
    if need_llm:
        from orion.llm.llm_backend import LLMBackend, LLMConfig
        from orion.llm.agent_b import AgentB
        from orion.llm.semantic_memory import SemanticMemory
        cfg = LLMConfig(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY",
                        model="default", temperature=0.05, max_tokens=2048)
        agent_b = AgentB(LLMBackend(cfg))
        kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
        if kb_path.exists():
            kb = SemanticMemory.from_json(kb_path)
            logger.info("K^B loaded: %d entries", len(kb.entries))
        else:
            logger.warning("K^B not found; LLM arms run without grounding")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    ckpt = out_dir / f"gate_colocation_{args.family}{tag}.json"
    commit, dirty = git_state()  # §O Δ3 — provenance in the result JSON
    if dirty:
        logger.warning("TREE IS DIRTY — §O Δ3 forbids firing the gate on an "
                       "uncommitted tree (launcher should have refused).")
    state = {"family": args.family, "seeds": args.seeds, "rounds": args.rounds,
             "arrivals": args.arrivals, "mu5": MU5, "plain_foc": PLAIN_FOC,
             "random_foc": RANDOM_FOC,
             "git_commit": commit, "git_dirty": dirty, "cells": {}}
    if ckpt.exists():
        try:
            state = json.load(open(ckpt)); state.setdefault("cells", {})
            logger.info("resumed: %d cells done", len(state["cells"]))
            # §O Δ3: a resumed run appends its current commit so cells that
            # span commits are visible in the record.
            state.setdefault("git_commits", [])
            if commit and commit not in state["git_commits"]:
                state["git_commits"].append(commit)
            state["git_dirty"] = dirty
        except Exception as e:
            logger.warning("checkpoint unreadable (%s), fresh start", e)

    def save():
        json.dump(state, open(ckpt, "w"), indent=2)

    delays = W.build_delays(R.generate_family_instance(FAM[args.family], seed=0))

    # Inline cost profiling (PREREG 2026-07-11 §M.4-Δ7): one power sampler for the whole run,
    # a per-cell collector. Captured DURING the experiment, raw per-decision, not just averages.
    prof_dir = out_dir / f"profiles{tag}"
    traces_dir = out_dir / f"traces{tag}"  # per-arrival behavioral traces (§N.2)
    ckpt_dir = out_dir / f"ckpt{tag}"      # §O.7 — permanent checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sampler = PowerSampler(hz=20.0)
    sampler.start()
    logger.info("profiling: gpu_energy=%s  cpu_energy=%s", sampler.gpu_available,
                "rapl-measured" if sampler.rapl_available else "tdp-estimate (no root)")

    for seed in args.seeds:
        need_bc = any(f"{a}|{seed}" not in state["cells"] for a in args.arms)
        bc_actors = None
        if need_bc:
            logger.info("### seed %d: BC-warm-starting shared frozen actors ###", seed)
            bc_actors = build_bc_actors(args.family, seed, args.bc_scenarios, args.bc_epochs)
        _, ceiling = R.compute_ceiling(R.generate_family_instance(FAM[args.family], seed=0), seed + 777)

        for arm in args.arms:
            key = f"{arm}|{seed}"
            if key in state["cells"]:
                continue
            t0 = time.time()
            coll = ProfileCollector(sampler, label=key)
            set_collector(coll)
            try:
                cell = _run_arm(arm, args, seed, bc_actors, agent_b, kb, ceiling, delays, ent_sched,
                                traces_dir=traces_dir, ckpt_dir=ckpt_dir)
            finally:
                set_collector(None)
            if cell is None:
                continue
            cell["profile"] = coll.save(prof_dir / f"{key.replace('|', '_')}.json")
            cell["cell_totals"] = coll.cell_totals()  # §O.9 per-cell totals
            # Behavioral trace to a sidecar (§N.2) — keeps the main results JSON lean.
            _trace = cell.pop("trace", None)
            if _trace is not None:
                traces_dir.mkdir(parents=True, exist_ok=True)
                json.dump(_trace, open(traces_dir / f"{key.replace('|', '_')}.json", "w"))
            state["cells"][key] = cell
            save()
            logger.info("[%s|seed%d] FoC=%.1f%% ent=%s (%.0fs)", arm, seed, cell["foc"],
                        f"{cell['entropy']:.3f}" if cell["entropy"] is not None else "n/a",
                        time.time() - t0)
    sampler.stop()

    # Aggregate mean +/- std across seeds.
    logger.info("\n" + "=" * 70)
    logger.info("GATE — colocation-prior contest  family=%s  (Plain=%.1f%%, random=%.1f%%)",
                args.family, PLAIN_FOC, RANDOM_FOC)
    logger.info("=" * 70)
    # §O Δ2: the aggregate (and every pre-named §N.4 reading) uses VALID cells
    # only. Learning cells with EV(tail-5) < 0.5 are invalid-execution, not
    # nulls, and are reported separately. Prior-only has no critic -> always
    # counted (criterion applies to learning cells only).
    agg = {}
    invalid = [k for k, c in state["cells"].items()
               if c.get("valid_execution") is False]
    if invalid:
        logger.warning("INVALID-EXECUTION cells (EV tail-5 < 0.5, §O Δ2): %s — "
                       "excluded from aggregate; NOT nulls.", invalid)
    state["invalid_execution_cells"] = invalid
    for arm in args.arms:
        cells = [state["cells"][f"{arm}|{s}"] for s in args.seeds
                 if f"{arm}|{s}" in state["cells"]
                 and state["cells"][f"{arm}|{s}"].get("valid_execution") is not False]
        focs = [c["foc"] for c in cells]
        ents = [c["entropy"] for c in cells if c["entropy"] is not None]
        if not focs:
            continue
        agg[arm] = {"mean": float(np.mean(focs)), "std": float(np.std(focs)), "n": len(focs),
                    "vs_random": float(np.mean(focs) - RANDOM_FOC),
                    "entropy": float(np.mean(ents)) if ents else None,
                    "ev_tail5": [c.get("ev_tail5") for c in cells]}
        logger.info("  %-11s FoC=%5.1f +-%-4.1f  (%+5.1f vs random)  ent=%s  n_valid=%d", arm,
                    agg[arm]["mean"], agg[arm]["std"], agg[arm]["vs_random"],
                    f"{agg[arm]['entropy']:.3f}" if agg[arm]["entropy"] is not None else "n/a",
                    len(focs))
    state["aggregate"] = agg

    # §O.9 — per-gate grand totals ("total cost to train and evaluate each arm").
    # Measured and estimated CPU energy are kept in SEPARATE fields, never mixed.
    grand = {"wall_s": 0.0, "cpu_s": 0.0, "gpu_energy_j": 0.0,
             "cpu_energy_j_measured": 0.0, "cpu_energy_j_estimated": 0.0}
    for cell in state["cells"].values():
        ct = cell.get("cell_totals") or {}
        grand["wall_s"] += ct.get("wall_s") or 0.0
        grand["cpu_s"] += ct.get("cpu_s") or 0.0
        grand["gpu_energy_j"] += ct.get("gpu_energy_j") or 0.0
        grand["cpu_energy_j_measured"] += ct.get("cpu_energy_j") or 0.0
        grand["cpu_energy_j_estimated"] += ct.get("cpu_energy_j_est") or 0.0
    state["gate_totals"] = grand
    logger.info("gate totals: wall=%.0fs cpu=%.0fs gpu=%.0fJ cpu_energy=%s",
                grand["wall_s"], grand["cpu_s"], grand["gpu_energy_j"],
                (f"{grand['cpu_energy_j_measured']:.0f}J measured"
                 if grand["cpu_energy_j_measured"]
                 else f"{grand['cpu_energy_j_estimated']:.0f}J ESTIMATED"))
    save()

    # Pre-named verdict.
    if all(a in agg for a in ("RL-alone", "LLM+RL", "Prior-only")):
        rl, llm, pri = agg["RL-alone"]["mean"], agg["LLM+RL"]["mean"], agg["Prior-only"]["mean"]
        logger.info("-" * 70)
        MARGIN = 3.0  # pts; below this, treat as "~="
        if llm > pri + MARGIN and llm > rl + MARGIN:
            logger.info("VERDICT: (2)>(3) AND (2)>(1) -> headline contest LIVE & honest. RL learns")
            logger.info("  useful deviations from the prior AND the prior matters.")
        elif abs(llm - pri) <= MARGIN:
            logger.info("VERDICT: (2)~=(3) -> MDO adds nothing over the prior at this scale.")
            logger.info("  Move the bottleneck hunt to credit assignment / observation BEFORE full-scale.")
        if rl <= RANDOM_FOC + MARGIN:
            logger.info("NOTE: (1)~=random even with BC+mu5 -> the RL problem itself may be broken;")
            logger.info("  no prior result is publishable on top of a broken base.")
        elif rl > RANDOM_FOC + 15:
            logger.info("NOTE: (1) lifts substantially -> the ACTOR hypothesis was the real story;")
            logger.info("  the 'RL can't beat random' framing dies (as it should).")
    logger.info("Saved: %s", ckpt)
    Path("runs").mkdir(exist_ok=True); Path(f"runs/GATE_COLOCATION{tag.upper()}_DONE").touch()


if __name__ == "__main__":
    main()
