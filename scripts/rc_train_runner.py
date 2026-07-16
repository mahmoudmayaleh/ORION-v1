"""§R local training experiment on the routing-critical family C+_T-_B-_RC.

Full training (not a 60-round gate) of the conformant §O stack on RC-v2:
  Full-ORION : LLaMA-3-8B Agent B plans (cache-ON) + PPO-trained MDO selector +
               BC-warm-started domain actors + annealed KL prior (beta 1->0).
  RL-alone   : greedy plans, no LLM, beta=0, same trained MDO + actors.

Both arms share the same BC actors; the arm difference is purely the LLM plan +
KL prior. Cache-ON collapses the LLM to ~6 signature calls on round 1, so many
rounds are feasible (the gate's 22h was per-arrival Agent B, no cache).

Readout order (committed): validity (EV_tail5 >= 0.5) -> behavioral trace
(selection entropy, reject taxonomy) -> FoC vs Plain-ColocFB 40.4 (RC-v2 validity)
and the pilot's local-follow_prior 44.4. RL selector must ADD admission beyond
plan-following to be valuable; RL-alone is its headroom without the LLM.

Run on the box (needs llama.cpp :8000 for the Full arm). --mock or --arm RL-alone
smoke the wiring with no server.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import wp7_runner as W
import five_arm_runner as R
from orion.substrate.routing_critical import (
    generate_rc_instance, rc_slice_factory, rc_topology_config,
    RC_GEN_SEED, RC_FAMILY_SHORT,
)
from orion.provenance import git_provenance, serving_provenance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rc_train")

RC_FAM = "C+_T-_B-_RC"
PLAIN_FOC_RC = 40.4       # RC-v2 validity draw (frozen reference)
PILOT_LOCAL_FOC = 44.4    # pilot ORION-local follow_prior (context)
BW_FOR_SEED = {42: 70.0, 43: 90.0, 44: 110.0}  # RC-v2 override sweep


def prereg_sha256() -> str:
    import hashlib
    p = Path(__file__).resolve().parent.parent / "docs" / "PREREG_AMENDMENT_2026-07-15_R.md"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def rc_substrate_fn(seed: int):
    bw = BW_FOR_SEED.get(seed, 70.0)
    return generate_rc_instance(RC_GEN_SEED + (seed - 42), inter_domain_bw_override=bw)


def build_rc_bc_actors(seed: int, scenarios: int, epochs: int):
    """BC-warm-start DomainActors on the RC-v2 topology, then FREEZE (§O choice)."""
    from orion.actors.domain_actor import DomainActor
    from orion.actors.policy import DomainPolicy
    from orion.training.bc_dataset import BCDatasetSpec
    from orion.training.bc_pretrain import bc_pretrain
    from orion.training.config import MAPPOConfig

    config = rc_topology_config()
    actors = {}
    for d in range(config.num_domains):
        torch.manual_seed(seed + 3_000_000 + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))
    bc_config = MAPPOConfig(bc_epochs=epochs, bc_lr=1e-3, bc_entropy_coef=0.01, bc_seed=seed)
    bc_spec = BCDatasetSpec(seed=seed, num_scenarios=scenarios, topology_config=config)
    t0 = time.time()
    bc_pretrain(domain_actors=actors, spec=bc_spec,
                dataset_path=Path(f"data/bc_demos_RC_{seed}.pt"), config=bc_config)
    logger.info("BC actors ready (seed=%d, %d scenarios, %d epochs) in %.1fs",
                seed, scenarios, epochs, time.time() - t0)
    return actors


def _tail(curve, key, n=5):
    vals = [c[key] for c in curve[-n:] if c.get(key) is not None and np.isfinite(c[key])]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--arrivals", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--actors", choices=["bc", "greedy"], default="bc")
    ap.add_argument("--bc-scenarios", type=int, default=2000)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--arm", choices=["both", "RL-alone", "Full-ORION"], default="both")
    ap.add_argument("--mock", action="store_true", help="FFD stand-in for the LLM (no server)")
    ap.add_argument("--tag", default="R")
    args = ap.parse_args()

    _prov = git_provenance(serving=serving_provenance(args.port), tag=args.tag,
                           prereg="docs/PREREG_AMENDMENT_2026-07-15_R.md")
    logger.info("provenance: commit=%s dirty=%s serving=%s",
                _prov["git_commit"][:8], _prov["git_dirty"], _prov["serving"])

    # Wire the RC family + cut-sensitive workload into the §O trainer.
    # §R Δ2-R: cache-OFF (registered configuration; the cache keys forced/flex chains
    # onto one sfc_template. The 'pathological' framing was withdrawn by §R Δ3-R),
    # fixed train stream (byte-identical every round = R.2's exact stream) + request_id
    # plan memo → per-arrival distinct plans at ~100 calls/seed, not 25k. R.4-vs-R.2 is
    # then exact same-stream selector isolation (also evaluated in-job via follow_prior).
    W.RC_SUBSTRATE_FN = rc_substrate_fn
    W.RC_SLICE_FACTORY = rc_slice_factory
    W.RC_USE_PLAN_CACHE = False
    W.RC_FIXED_TRAIN_STREAM = True

    agent_b, kb = None, None
    want_full = args.arm in ("both", "Full-ORION") and not args.mock
    if want_full:
        from orion.llm.llm_backend import LLMBackend, LLMConfig
        from orion.llm.agent_b import AgentB
        from orion.llm.semantic_memory import SemanticMemory
        cfg = LLMConfig(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY",
                        model="default", temperature=0.0, max_tokens=2048)
        agent_b = AgentB(LLMBackend(cfg))
        kb_path = Path(__file__).resolve().parent.parent / "data" / "kb_entries.json"
        if kb_path.exists():
            kb = SemanticMemory.from_json(kb_path)
            logger.info("K^B loaded: %d entries", len(kb.entries))

    ckpt_dir = Path("results/wp7/ckpt_RC")
    out = {"provenance": _prov, "tag": args.tag, "family": RC_FAM, "gen_seed": RC_GEN_SEED,
           "prereg_sha256": prereg_sha256(),
           "cache_off": not W.RC_USE_PLAN_CACHE, "fixed_train_stream": "Full-ORION only (RL-alone varied)",
           "rounds": args.rounds, "arrivals": args.arrivals, "seeds": args.seeds,
           "actors": args.actors, "plain_foc": PLAIN_FOC_RC, "cells": {}}

    for seed in args.seeds:
        actors = None if args.actors == "greedy" else build_rc_bc_actors(
            seed, args.bc_scenarios, args.bc_epochs)

        if args.arm in ("both", "RL-alone"):
            # RL-alone has no LLM to memoize → use varied per-round streams (its
            # validated SMOKE100 protocol, more coverage). Fixed stream is only for
            # the Full-ORION plan memo.
            W.RC_FIXED_TRAIN_STREAM = False
            logger.info("### RL-alone seed=%d rounds=%d", seed, args.rounds)
            curve = W.train_arm("RL-alone", RC_FAM, seed, args.rounds, args.arrivals,
                                args.lr, 0.0, 0.0, None, None, mock=True, actors=actors,
                                entropy_schedule=(0.03, 0.01),
                                ckpt_path=str(ckpt_dir / f"RL-alone_{seed}.pt"))
            out["cells"][f"RL-alone|{seed}"] = _cell(curve)

        if args.arm in ("both", "Full-ORION") and (want_full or args.mock):
            W.RC_FIXED_TRAIN_STREAM = True  # fixed stream = R.2's exact stream + plan memo
            logger.info("### Full-ORION seed=%d rounds=%d mock=%s", seed, args.rounds, args.mock)
            curve = W.train_arm("LLM+RL-full", RC_FAM, seed, args.rounds, args.arrivals,
                                args.lr, 1.0, 0.0, agent_b, kb, mock=args.mock, actors=actors,
                                entropy_schedule=(0.03, 0.01), eval_with_train_builder=True,
                                ckpt_path=str(ckpt_dir / f"Full-ORION_{seed}.pt"))
            out["cells"][f"Full-ORION|{seed}"] = _cell(curve)

    out_path = Path(f"data/rc_train_results_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("results -> %s", out_path)
    _readout(out)


def _cell(curve):
    last = curve[-1] if curve else {}
    foc_follow = last.get("eval_foc_follow_prior")
    foc_trained = last.get("eval_foc_trained", last.get("eval_foc"))
    return {
        "ev_tail5": _tail(curve, "ev", 5),
        "foc_plateau": 100.0 * _tail(curve, "eval_foc", 10),
        "entropy_tail5": _tail(curve, "mdo_entropy", 5),
        "mtilde_agreement_tail5": _tail(curve, "mtilde_agreement", 5),
        "final_foc": 100.0 * (curve[-1]["eval_foc"] if curve else float("nan")),
        # §R Δ2-R same-stream selector isolation (held-out seed+777, identical plans):
        "foc_trained": 100.0 * foc_trained if foc_trained is not None else float("nan"),
        "foc_follow_prior": 100.0 * foc_follow if foc_follow is not None else float("nan"),
        "selector_delta_pp": (100.0 * (foc_trained - foc_follow)
                              if (foc_trained is not None and foc_follow is not None)
                              else float("nan")),
        "rounds": len(curve),
        "curve": curve,
    }


def _readout(out):
    print("\n" + "=" * 74)
    print(f"RC TRAINING READOUT — {out['family']} (rounds={out['rounds']}, "
          f"arrivals={out['arrivals']}, actors={out['actors']})")
    print("=" * 74)
    cells = out["cells"]

    print("\n[1] VALIDITY (EV_tail5 >= 0.5 required; else cell INVALID-EXECUTION)")
    for k, c in cells.items():
        ok = "VALID" if (c["ev_tail5"] >= 0.5) else "INVALID"
        print(f"  {k:20s} EV_tail5={c['ev_tail5']:.3f}  [{ok}]")

    print("\n[2] BEHAVIORAL (selection entropy, m~-agreement — tail5)")
    for k, c in cells.items():
        print(f"  {k:20s} entropy={c['entropy_tail5']:.3f}  "
              f"m~agree={c['mtilde_agreement_tail5'] if c['mtilde_agreement_tail5']==c['mtilde_agreement_tail5'] else float('nan'):.2f}")

    print(f"\n[3] FoC plateau (last-10 mean) — refs: Plain {PLAIN_FOC_RC}%, "
          f"pilot local-follow_prior {PILOT_LOCAL_FOC}%")
    for k, c in cells.items():
        vs_plain = c["foc_plateau"] - PLAIN_FOC_RC
        print(f"  {k:20s} FoC={c['foc_plateau']:.1f}%  (vs Plain {vs_plain:+.1f}pp)  "
              f"[final={c['final_foc']:.1f}%]")

    # §R Δ2-R — SELECTOR ISOLATION: trained MDO vs follow_prior on identical held-out
    # streams + plans (only the selector differs). This is the honest R.2-equivalent
    # comparator (same plan source, same sampling), replacing the pilot 44.4 baseline.
    print("\n[4] SELECTOR ISOLATION (held-out seed+777, identical plans, only selector differs)")
    for k, c in cells.items():
        print(f"  {k:20s} trained={c['foc_trained']:.1f}%  follow_prior={c['foc_follow_prior']:.1f}%  "
              f"selector_delta={c['selector_delta_pp']:+.1f}pp")
    print("  (selector_delta > 0 => the trained MDO adds admission BEYOND following the plan)")

    # Pre-named comparisons (per seed sign where both arms present)
    seeds = sorted({k.split("|")[1] for k in cells})
    print("\n[pre-named comparisons]")
    for s in seeds:
        full = cells.get(f"Full-ORION|{s}")
        rl = cells.get(f"RL-alone|{s}")
        if full and rl:
            print(f"  seed {s}: Full={full['foc_plateau']:.1f}%  RL-alone={rl['foc_plateau']:.1f}%  "
                  f"Full-RL={full['foc_plateau']-rl['foc_plateau']:+.1f}pp  "
                  f"RL-Plain={rl['foc_plateau']-PLAIN_FOC_RC:+.1f}pp")


if __name__ == "__main__":
    main()
