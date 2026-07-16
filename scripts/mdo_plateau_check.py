#!/usr/bin/env python3
"""MDO plateau check on C+_T-_B- — does a longer, properly-diagnosed MDO run lift
off the ~20% Probe-A floor, and if not, WHICH component caps it?

NOTE on "proper trainer": the codebase's MDO update (readiness_test.py) uses a
CONTEXTUAL-BANDIT advantage (A = R - V(s)), deliberately, because each arrival is an
independent decision (comment at readiness_test.py:221-223). GAE-across-arrivals is not
the intended formulation. So this uses R - V(s) too; the payload is the DIAGNOSTIC
BASELINES that localize the ceiling, not a trainer swap.

Frozen GreedyDomainActor (isolates the MDO; matches Probe A's greedy condition). RL-alone
(beta=0, no LLM). Per-round deterministic eval FoC vs exhaustive ceiling + MDO entropy.
Final baselines on K identical eval streams:
  learned(det)  : the trained MDO
  greedy(prior) : follow_prior on the greedy FFD partition (the MDO's own prior)
  random(mask)  : uniform over tier-feasible domains (learned-vs-random floor)
  pure(a=0)     : a FRESH MDO trained with alpha=xi=eta=0 (admission-only reward)
Reads:
  learned >> random  -> MDO learned something beyond feasibility
  learned << greedy  -> MDO partition worse than its greedy prior (the real gap)
  pure  >  learned   -> cost/trial terms are suppressing admission -> reward shaping
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
from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.observation import build_mdo_observation, observation_to_tensor
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.reward import RewardWeights
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.critic import CentralisedCritic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plateau")

ARRIVAL_RATE = R.ARRIVAL_RATE
SERVICE_RATE = R.SERVICE_RATE
FAM = {f.short_name: f for f in R.ALL_FAMILIES}


def build_delays(sub):
    delays = {}
    g = sub.graph
    for u, v, d in g.edges(data=True):
        sd, dd = g.nodes[u]["domain_id"], g.nodes[v]["domain_id"]
        if sd != dd:
            delays[(min(sd, dd), max(sd, dd))] = min(
                delays.get((min(sd, dd), max(sd, dd)), 1e9), d["propagation_delay"])
    return delays


def greedy_plan_builder(slice_req, substrate):
    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
    if not result.feasible or result.plan is None:
        return None
    return R.plan_to_summary(result, slice_req, substrate)


def obs_dim_for(sub):
    dummy = PlanSummary(vnf_ids=["v0", "v1"], required_tiers=[R.InfrastructureTier.MEC] * 2,
                        suggested_domains=[0, 1], cpu_demands=[1.0] * 2, ram_demands=[1.0] * 2,
                        vcrs=[1.0] * 2, bw_demands=[10.0])
    return observation_to_tensor(build_mdo_observation(sub, dummy), max_vnfs=10).shape[0]


def train_mdo(family, seed, rounds, arrivals, lr, mdo_cfg, eval_ceiling, delays, label):
    """Train an MDO (frozen greedy actors), return (policy, coord, curve)."""
    torch.manual_seed(seed); np.random.seed(seed)
    sub0 = R.generate_family_instance(FAM[family], seed=0)
    num_domains = sub0.num_domains
    odim = obs_dim_for(sub0)
    policy = MDOPolicy(obs_dim=odim, num_domains=num_domains, max_vnfs=10, hidden_dim=128, num_layers=2)
    actors = {d: GreedyDomainActor(d) for d in range(num_domains)}  # frozen (no params)
    coord = MDOCoordinator(policy, actors, mdo_cfg)
    critic = CentralisedCritic(input_dim=odim, hidden_dim=128, num_layers=2)
    opt_mdo = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_crit = torch.optim.Adam(critic.parameters(), lr=3e-4)
    cfg = MAPPOConfig(update_epochs=4, clip_eps=0.2, entropy_coef=0.01, gamma=0.99, gae_lambda=0.95)
    rw = RewardWeights(lambda_viol=10.0)
    curve = []
    for rnd in range(rounds):
        t0 = time.time()
        sub = R.generate_family_instance(FAM[family], seed=0)
        rng = np.random.default_rng(seed + rnd * 1_000_000 + 1)
        ap = ArrivalProcess(sub, arrivals, ARRIVAL_RATE, SERVICE_RATE, rng); ap.generate()
        runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=greedy_plan_builder, reward_weights=rw)
        runner.reset()
        ep = runner.run_episode(mdo_mode="sample")
        buf = PPORolloutBuffer()
        with torch.no_grad():
            for t in ep.rollout.mdo:
                oc = t.obs if t.obs.numel() > 0 else torch.zeros(odim)
                buf.append_mdo(mdo_obs=t.obs, action=t.action, log_prob=t.log_probs, entropy=t.entropy,
                               aux_value=t.value_estimate, global_state=oc,
                               critic_value=float(critic(oc.unsqueeze(0)).item()),
                               reward=t.terminal_reward, done=t.committed,
                               tier_mask=t.tier_mask, num_vnfs=t.num_vnfs)
        ent = float(np.mean([float(t.entropy) for t in ep.rollout.mdo])) if ep.rollout.mdo else 0.0
        if len(buf) > 0:
            rewards = buf.reward_tensor()
            with torch.no_grad():
                vals = torch.tensor([float(critic(o.unsqueeze(0)).item()) if o.numel() > 0 else 0.0
                                     for o in buf.mdo_obs], dtype=torch.float32)
            advantages = rewards - vals  # contextual-bandit advantage (codebase design)
            returns = rewards.clone(); buf.set_gae(advantages, returns)
            for _ in range(cfg.update_epochs):  # critic
                gs = torch.stack(buf.global_states)
                old_v = torch.tensor(buf.critic_values, dtype=torch.float32)
                nv = critic(gs).squeeze(-1)
                vc = old_v + torch.clamp(nv - old_v, -cfg.clip_eps, cfg.clip_eps)
                vl = 0.5 * torch.max((nv - returns) ** 2, (vc - returns) ** 2).mean()
                opt_crit.zero_grad(); (0.5 * vl).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt_crit.step()
            for _ in range(cfg.update_epochs):  # MDO PPO
                el = torch.tensor(0.0); ec = 0
                for i in range(len(buf.mdo_obs)):
                    oi = buf.mdo_obs[i]
                    if oi.numel() == 0:
                        continue
                    k = buf.mdo_num_vnfs[i]
                    if k == 0 or i >= len(buf.mdo_tier_masks):
                        continue
                    ai = torch.tensor(buf.mdo_actions[i], dtype=torch.long)
                    tm = buf.mdo_tier_masks[i]
                    adv = float(advantages[i]) if i < len(advantages) else 0.0
                    nlp, ne, _ = policy.evaluate_actions(oi, tm, ai, k)
                    ratio = torch.exp(nlp.sum() - buf.mdo_log_probs[i].sum())
                    el = el + (-torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)) \
                        - cfg.entropy_coef * ne
                    ec += 1
                if ec > 0:
                    opt_mdo.zero_grad(); (el / ec).backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt_mdo.step()
        foc = eval_det(coord, family, seed, arrivals, eval_ceiling, delays)
        curve.append({"round": rnd + 1, "eval_foc": foc, "train_admit": ep.stats.admitted,
                      "train_total": ep.stats.total_arrivals, "mdo_entropy": ent})
        if (rnd + 1) % 10 == 0 or rnd < 3:
            logger.info("[%s] R%d: train=%d/%d eval_FoC=%.1f%% ent=%.3f %.1fs", label, rnd + 1,
                        ep.stats.admitted, ep.stats.total_arrivals, 100 * foc, ent, time.time() - t0)
    return policy, coord, curve


def eval_det(coord, family, seed, arrivals, ceiling, delays, mode="deterministic"):
    sub = R.generate_family_instance(FAM[family], seed=0)
    rng = np.random.default_rng(seed + 777)
    ap = ArrivalProcess(sub, arrivals, ARRIVAL_RATE, SERVICE_RATE, rng); ap.generate()
    runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=greedy_plan_builder,
                           reward_weights=RewardWeights(lambda_viol=10.0))
    runner.reset()
    ep = runner.run_episode(mdo_mode=mode)
    return ep.stats.admitted / ceiling if ceiling > 0 else 0.0


def baseline_foc(family, seed, arrivals, ceiling, delays, mode, policy=None):
    """FoC of a fixed policy/mode (greedy prior / random) with frozen greedy actors."""
    sub0 = R.generate_family_instance(FAM[family], seed=0)
    actors = {d: GreedyDomainActor(d) for d in range(sub0.num_domains)}
    coord = MDOCoordinator(policy, actors,
                           MDOConfig(n_part=(1 if mode == "follow_prior" else 3),
                                     mu=1.0, alpha=0.1, xi=0.1, eta=0.1))
    return eval_det(coord, family, seed, arrivals, ceiling, delays, mode=mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="C+_T-_B-")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--arrivals", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="results/wp7")
    args = ap.parse_args()

    delays = build_delays(R.generate_family_instance(FAM[args.family], seed=0))
    _, ceiling = R.compute_ceiling(R.generate_family_instance(FAM[args.family], seed=0), args.seed + 777)
    logger.info("MDO plateau check: family=%s seed=%d rounds=%d ceiling=%d",
                args.family, args.seed, args.rounds, ceiling)

    # ORION reward (mu admission + alpha cost + xi trial + eta quality)
    _, _, curve = train_mdo(args.family, args.seed, args.rounds, args.arrivals, args.lr,
                            MDOConfig(n_part=3, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
                            ceiling, delays, "orion")
    learned = np.mean([c["eval_foc"] for c in curve[-10:]]) * 100
    ent_last = np.mean([c["mdo_entropy"] for c in curve[-10:]])

    # Admission-pure reward (alpha=xi=eta=0): does removing cost/trial terms lift admission?
    logger.info("--- admission-pure ablation (alpha=xi=eta=0) ---")
    _, _, curve_pure = train_mdo(args.family, args.seed, args.rounds, args.arrivals, args.lr,
                                 MDOConfig(n_part=3, mu=1.0, alpha=0.0, xi=0.0, eta=0.0),
                                 ceiling, delays, "pure")
    pure = np.mean([c["eval_foc"] for c in curve_pure[-10:]]) * 100

    # Fixed-policy baselines.
    greedy_prior = 100 * baseline_foc(args.family, args.seed, args.arrivals, ceiling, delays, "follow_prior", None)
    random_mask = 100 * baseline_foc(args.family, args.seed, args.arrivals, ceiling, delays, "random", None)

    out = {"family": args.family, "seed": args.seed, "rounds": args.rounds, "ceiling": ceiling,
           "learned_foc": float(learned), "learned_entropy_last10": float(ent_last),
           "pure_foc": float(pure), "greedy_prior_foc": float(greedy_prior),
           "random_foc": float(random_mask), "probe_a_greedy_plateau": 17.2,
           "curve_orion": curve, "curve_pure": curve_pure}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"mdo_plateau_{args.family}_seed{args.seed}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("\n" + "=" * 64)
    logger.info("MDO PLATEAU CHECK  family=%s  (ceiling=%d)", args.family, ceiling)
    logger.info("=" * 64)
    logger.info("  learned MDO (ORION reward, last-10) = %.1f%%  (entropy %.3f)", learned, ent_last)
    logger.info("  admission-pure MDO (alpha=xi=eta=0)  = %.1f%%", pure)
    logger.info("  greedy prior (follow_prior)          = %.1f%%   <- the MDO's own prior", greedy_prior)
    logger.info("  masked-random (no policy)            = %.1f%%", random_mask)
    logger.info("  Probe A greedy plateau (80 rds prior)= 17.2%%")
    logger.info("  reads:")
    logger.info("    learned vs random  = %+.1f pts (>0 => learned beyond feasibility)", learned - random_mask)
    logger.info("    learned vs greedy  = %+.1f pts (<0 => MDO worse than its greedy prior)", learned - greedy_prior)
    logger.info("    pure vs learned    = %+.1f pts (>0 => cost/trial terms suppress admission)", pure - learned)
    logger.info("Saved: %s", p)
    Path("runs").mkdir(exist_ok=True); Path("runs/MDO_PLATEAU_DONE").touch()


if __name__ == "__main__":
    main()
