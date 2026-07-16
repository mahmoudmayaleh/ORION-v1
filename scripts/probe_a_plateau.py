#!/usr/bin/env python3
"""Probe A — BC-frozen-actor plateau (decides: actors-are-ceiling vs MDO-is-ceiling).

Question: is the WP7 ~20% MDO plateau caused by the frozen placer, or by the MDO
partition quality / reward / n_part? Run RL-alone (beta=0, NO LLM) on C+_T-_B-
(the pre-registered headline family) long enough to plateau, under TWO frozen
placers, and compare plateau height to Plain-ColocFB's 85.3%:

  greedy : GreedyDomainActor (best-fit heuristic; the WP7 stand-in)
  bc     : BC-warm-started DomainActor, frozen (the actual full-scale placer)

If BOTH plateau near ~20% << 85%, the placer is NOT the ceiling -> the MDO
partition/reward/n_part is; more rounds/seeds replicate a floor. If bc lifts the
plateau toward Plain, the placer was the ceiling -> full-scale is warranted.

Read-outs: eval FoC curve (vs exhaustive ceiling), samples-to-threshold, AND MDO
policy ENTROPY per round (stable-vs-trained: is the policy still exploring, or
frozen? A lift with collapsed entropy is a different story than a lift with
healthy exploration).
"""
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

import five_arm_runner as R
import wp7_runner as W
from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.substrate.topology_families import family_to_config
from orion.training.bc_dataset import BCDatasetSpec
from orion.training.bc_pretrain import bc_pretrain
from orion.training.config import MAPPOConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_a")

HIDDEN_DIM = 64
PLAIN_FOC = 85.3  # C+_T-_B- Plain-ColocFB FoC (results/wp7/PROBE_HEADROOM_RESULT.md)


def build_bc_actors(family, seed, scenarios, epochs):
    """BC-warm-start DomainActors on the family's topology config, then FREEZE."""
    fam = {f.short_name: f for f in R.ALL_FAMILIES}[family]
    config = family_to_config(fam)  # 5 domains, C+ caps, B- links (tier split is
    # runtime post-processing; the action mask enforces tier feasibility at eval)
    actors = {}
    for d in range(config.num_domains):
        torch.manual_seed(seed + 3_000_000 + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))

    bc_config = MAPPOConfig(bc_epochs=epochs, bc_lr=1e-3, bc_entropy_coef=0.01, bc_seed=seed)
    bc_spec = BCDatasetSpec(seed=seed, num_scenarios=scenarios, topology_config=config)
    logger.info("BC pretraining: %d scenarios, %d epochs (family=%s, %d domains)...",
                scenarios, epochs, family, config.num_domains)
    t0 = time.time()
    logs, meta = bc_pretrain(domain_actors=actors, spec=bc_spec,
                             dataset_path=Path(f"data/bc_demos_{family}.pt"), config=bc_config)
    logger.info("BC done in %.1fs (dataset hash %s)", time.time() - t0, meta.get("dataset_hash"))
    for d, epoch_logs in logs.items():
        if epoch_logs:
            last = epoch_logs[-1]
            logger.info("  domain %d: final BC loss=%.4f entropy=%.4f samples=%d",
                        d, last.imitation_loss, last.entropy_bonus, last.num_samples)
    # FREEZE
    for a in actors.values():
        for p in a.policy.parameters():
            p.requires_grad = False
    return actors


def summarize(name, curve):
    foc = np.array([c["eval_foc"] for c in curve]) * 100
    ent = np.array([c["mdo_entropy"] for c in curve])
    last10 = foc[-10:].mean()
    # samples-to-threshold: cumulative train arrivals to first reach 90% of plateau
    target = 0.9 * last10
    hit = next((c["cumulative_arrivals"] for c in curve if c["eval_foc"] * 100 >= target), None)
    logger.info("[%s] plateau(last10)=%.1f%%  max=%.1f%%  final_entropy=%.3f  "
                "entropy(last10 mean)=%.3f  samples_to_90%%=%s",
                name, last10, foc.max(), ent[-1], ent[-10:].mean(), hit)
    return {"plateau_last10": float(last10), "max": float(foc.max()),
            "final_entropy": float(ent[-1]), "entropy_last10": float(ent[-10:].mean()),
            "samples_to_90pct": hit, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="C+_T-_B-")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--arrivals", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bc-scenarios", type=int, default=2500)
    ap.add_argument("--bc-epochs", type=int, default=8)
    ap.add_argument("--conditions", nargs="+", default=["greedy", "bc"])
    ap.add_argument("--out", default="results/wp7")
    args = ap.parse_args()

    results = {}
    # Greedy placer condition (no BC).
    if "greedy" in args.conditions:
        logger.info("=== CONDITION greedy (GreedyDomainActor) ===")
        curve = W.train_arm("RL-alone", args.family, args.seed, args.rounds, args.arrivals,
                            args.lr, 0.0, 0.0, None, None, True, actors=None)
        results["greedy"] = summarize("greedy", curve)

    # BC-frozen DomainActor condition.
    if "bc" in args.conditions:
        logger.info("=== CONDITION bc (BC-frozen DomainActor) ===")
        bc_actors = build_bc_actors(args.family, args.seed, args.bc_scenarios, args.bc_epochs)
        curve = W.train_arm("RL-alone", args.family, args.seed, args.rounds, args.arrivals,
                            args.lr, 0.0, 0.0, None, None, True, actors=bc_actors)
        results["bc"] = summarize("bc", curve)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_a_{args.family}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"family": args.family, "seed": args.seed, "plain_foc": PLAIN_FOC,
                   "results": {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                               for k, v in results.items()},
                   "curves": {k: v["curve"] for k, v in results.items()}}, f, indent=2)

    logger.info("\n" + "=" * 66)
    logger.info("PROBE A VERDICT  family=%s  (Plain-ColocFB = %.1f%%)", args.family, PLAIN_FOC)
    logger.info("=" * 66)
    for k, v in results.items():
        gap = PLAIN_FOC - v["plateau_last10"]
        logger.info("  %-7s plateau=%.1f%%  (%.1f pts below Plain)  entropy=%.3f",
                    k, v["plateau_last10"], gap, v["entropy_last10"])
    if "bc" in results and "greedy" in results:
        lift = results["bc"]["plateau_last10"] - results["greedy"]["plateau_last10"]
        logger.info("  BC lift over greedy = %+.1f pts", lift)
        best = max(results.values(), key=lambda v: v["plateau_last10"])["plateau_last10"]
        if best >= PLAIN_FOC - 15:
            logger.info("  -> placer materially lifts plateau toward Plain: actors were the")
            logger.info("     ceiling. Full-scale warranted; headline contest is live.")
        else:
            logger.info("  -> plateau stays far below Plain under BOTH placers: the placer is")
            logger.info("     NOT the ceiling. Bottleneck is MDO partition/reward/n_part.")
            logger.info("     Full-scale would replicate a floor -> debug the MDO first.")
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
