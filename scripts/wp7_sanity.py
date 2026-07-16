#!/usr/bin/env python3
"""WP7 tiny-instance sanity: confirm the MDO learning loop fires end-to-end
before any full-scale launch.

Reuses the ablation MDO-only harness (MDOPolicy + frozen GreedyDomainActor +
MDOCoordinator n_part=3 + CentralisedCritic + KL-prior PPO update via
EpisodeRunner.run_episode). Runs TWO conditions on ONE hardest family
(C-_T-_B-) with ~50 arrivals, a few rounds:

  RL-alone   : β=0  (no KL prior, LLM-free m̃ ignored)     -> KL must NOT fire
  LLM+RL     : β anneal 1.0->0.3 (KL prior toward greedy m̃) -> KL MUST fire

Confirms commit / reject / reward(PPO param motion) / KL all fire. m̃ here is
the greedy structural partition (stand-in for Agent B for the smoke test; the
KL machinery is identical when m̃ comes from the real Agent B plan builder).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ablation_anneal_vs_beta0 as A
import five_arm_runner as R
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig

# ── Shrink to a tiny instance ────────────────────────────────────────────────
# Mechanism smoke test only, so we use the ablation's native 3-domain substrate
# (the MDOPolicy in run_condition is hardcoded to num_domains=3). The reduced-
# scale ON-CLAIM run on C-_T-_B- (5 domains) needs num_domains wired from the
# substrate — a one-line change in the eventual wp7 runner, NOT here.
A.NUM_ARRIVALS = 50


def _robust_greedy_plan_builder(slice_req, substrate):
    # domain_id read from the graph (not string-parsed) so it is substrate-
    # agnostic; reuses the production plan->summary conversion.
    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
    if not result.feasible or result.plan is None:
        return None
    return R.plan_to_summary(result, slice_req, substrate)


A._greedy_plan_builder = _robust_greedy_plan_builder


def summarize(name, res):
    admits = sum(r["admitted"] for r in res)
    totals = sum(r["total"] for r in res)
    rejects = totals - admits
    kl_max = max((r["kl"] for r in res), default=0.0)
    betas = [round(r["beta"], 3) for r in res]
    motion = res[-1]["param_motion"] if res else 0.0
    print(f"\n[{name}] rounds={len(res)} admit={admits}/{totals} reject={rejects} "
          f"kl_max={kl_max:.4f} beta={betas} param_motion={motion:.4f}")
    return {"admits": admits, "rejects": rejects, "kl_max": kl_max, "motion": motion}


if __name__ == "__main__":
    print(f"WP7 sanity: substrate=generic-3-domain  arrivals={A.NUM_ARRIVALS}  rounds=3")

    rl_alone = A.run_condition("RL-alone(b0)", seed=42, rounds=3,
                               kl_beta_start=0.0, kl_beta_end=0.0)
    llm_rl = A.run_condition("LLM+RL(anneal)", seed=42, rounds=3,
                             kl_beta_start=1.0, kl_beta_end=0.3)

    s0 = summarize("RL-alone", rl_alone)
    s1 = summarize("LLM+RL", llm_rl)

    print("\n=== SANITY VERDICT ===")
    checks = {
        "commit fires (both arms admit >0)": s0["admits"] > 0 and s1["admits"] > 0,
        "reject fires (both arms reject >0)": s0["rejects"] > 0 and s1["rejects"] > 0,
        "reward/PPO fires (policy params move)": s0["motion"] > 0 and s1["motion"] > 0,
        "KL OFF at b0 (RL-alone kl==0)": s0["kl_max"] == 0.0,
        "KL ON at anneal (LLM+RL kl>0)": s1["kl_max"] > 0.0,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\nRESULT:", "PASS" if all(checks.values()) else "NEEDS REVIEW")
