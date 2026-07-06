#!/usr/bin/env python3
"""Ablation: annealed-β vs β0 — does Agent B's prior improve MDO partition quality?

Both conditions use greedy-FFD fixed actors to isolate the MDO partition
signal from actor placement quality. The only variable is the KL prior:
  β0:     kl_beta = 0 (no prior, uniform initialisation)
  anneal: kl_beta decays linearly from β_start to β_end over all rounds

Usage:
    python scripts/ablation_anneal_vs_beta0.py --seed 0 --rounds 40
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.greedy_domain_actor import GreedyDomainActor
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.kl_prior import analytical_kl, build_prior_logits, beta_schedule
from orion.mdo.observation import build_mdo_observation, observation_to_tensor
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.critic import CentralisedCritic
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation")

NUM_ARRIVALS = 200
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02


def build_substrate(seed):
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(num_domains=3, nodes_per_domain=[8, 10, 12],
                       intra_link_density=0.4, inter_domain_links=4), rng)


def build_delays(substrate):
    delays = {}
    for u, v, d in substrate.graph.edges(data=True):
        sd = substrate.graph.nodes[u]["domain_id"]
        dd = substrate.graph.nodes[v]["domain_id"]
        if sd != dd:
            delays[(sd, dd)] = min(delays.get((sd, dd), float("inf")), d["propagation_delay"])
    return delays


def _greedy_plan_builder(slice_req, substrate):
    result = _run_greedy_ffd(substrate, slice_req, GreedyConfig())
    if not result.feasible or result.plan is None:
        return None
    vnf_ids, required_tiers, suggested_domains = [], [], []
    for vnf in slice_req.vnfs:
        node_id = result.plan.vnf_placements[vnf.vnf_id]
        domain = int(node_id.split("n")[0][1:])
        tier_str = substrate.graph.nodes[node_id]["tier"]
        vnf_ids.append(vnf.vnf_id)
        required_tiers.append(InfrastructureTier(tier_str))
        suggested_domains.append(domain)
    return PlanSummary(
        vnf_ids=vnf_ids, required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


def run_condition(
    name: str,
    seed: int,
    rounds: int,
    kl_beta_start: float,
    kl_beta_end: float,
    lr: float = 3e-3,
):
    """Run one ablation condition and return per-round stats."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    substrate = build_substrate(seed)
    delays = build_delays(substrate)

    max_vnfs = 10
    num_domains = 3
    dummy_plan = PlanSummary(
        vnf_ids=["v0", "v1"], required_tiers=[InfrastructureTier.MEC] * 2,
        suggested_domains=[0, 1], cpu_demands=[1.0] * 2, ram_demands=[1.0] * 2,
        vcrs=[1.0] * 2, bw_demands=[10.0],
    )
    obs = build_mdo_observation(substrate, dummy_plan)
    obs_tensor = observation_to_tensor(obs, max_vnfs=max_vnfs)
    obs_dim = obs_tensor.shape[0]

    mdo_policy = MDOPolicy(
        obs_dim=obs_dim, num_domains=num_domains,
        max_vnfs=max_vnfs, hidden_dim=128, num_layers=2,
    )

    actors = {d: GreedyDomainActor(d) for d in range(num_domains)}

    coord = MDOCoordinator(mdo_policy, actors, MDOConfig(
        n_part=3, mu=1.0, alpha=0.1, xi=0.1, eta=0.1,
    ))

    critic = CentralisedCritic(input_dim=obs_dim, hidden_dim=128, num_layers=2)

    mdo_optimizer = torch.optim.Adam(mdo_policy.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    cfg = MAPPOConfig(
        kl_beta_initial=kl_beta_start,
        kl_beta_final=kl_beta_end,
        update_epochs=4, clip_eps=0.2, entropy_coef=0.01,
        gamma=0.99, gae_lambda=0.95,
    )

    initial_params = {n: p.clone() for n, p in mdo_policy.named_parameters()}
    total_steps = 0
    total_decay_steps = rounds * NUM_ARRIVALS

    results = []

    logger.info("=" * 60)
    logger.info("CONDITION: %s  (β_start=%.2f β_end=%.2f lr=%.0e)", name, kl_beta_start, kl_beta_end, lr)
    logger.info("=" * 60)

    for rnd in range(rounds):
        t0 = time.time()

        substrate = build_substrate(seed)
        rng = np.random.default_rng(seed + rnd * 1_000_000 + 1)
        ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coord, delays, plan_builder=_greedy_plan_builder)
        runner.reset()
        ep = runner.run_episode(mdo_mode="sample")

        admitted = ep.stats.admitted
        total = ep.stats.total_arrivals
        rate = 100 * admitted / total if total > 0 else 0

        buffer = PPORolloutBuffer()
        suggested_domains_list = []
        with torch.no_grad():
            for t in ep.rollout.mdo:
                obs_for_critic = t.obs if t.obs.numel() > 0 else torch.zeros(obs_dim)
                cv = float(critic(obs_for_critic.unsqueeze(0)).item())
                buffer.append_mdo(
                    mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                    entropy=t.entropy, aux_value=t.value_estimate,
                    global_state=obs_for_critic, critic_value=cv,
                    reward=t.terminal_reward, done=t.committed,
                    tier_mask=t.tier_mask, num_vnfs=t.num_vnfs,
                )
                suggested_domains_list.append(t.info.get("suggested_domains", []))

        if len(buffer) == 0:
            logger.info("Round %d: empty buffer, skipping", rnd)
            continue

        rewards = buffer.reward_tensor()
        with torch.no_grad():
            values_list = []
            for obs_i in buffer.mdo_obs:
                if obs_i.numel() > 0:
                    values_list.append(float(critic(obs_i.unsqueeze(0)).item()))
                else:
                    values_list.append(0.0)
            values_t = torch.tensor(values_list, dtype=torch.float32)
        advantages = rewards - values_t
        returns = rewards.clone()
        buffer.set_gae(advantages, returns)

        # Critic update
        critic_loss_val = 0.0
        for _epoch in range(cfg.update_epochs):
            gs = torch.stack(buffer.global_states)
            ret = returns.detach()
            old_v = torch.tensor(buffer.critic_values, dtype=torch.float32)
            new_v = critic(gs).squeeze(-1)
            v_clipped = old_v + torch.clamp(new_v - old_v, -cfg.clip_eps, cfg.clip_eps)
            raw_loss = (new_v - ret) ** 2
            clip_loss = (v_clipped - ret) ** 2
            v_loss = 0.5 * torch.max(raw_loss, clip_loss).mean()
            critic_optimizer.zero_grad()
            (0.5 * v_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
            critic_loss_val += v_loss.item()

        # MDO PPO update with optional KL prior
        mdo_policy_loss = 0.0
        mdo_entropy_sum = 0.0
        mdo_steps = 0
        mdo_clips = 0
        kl_sum = 0.0

        for _epoch in range(cfg.update_epochs):
            epoch_loss = torch.tensor(0.0)
            epoch_count = 0

            for i in range(len(buffer.mdo_obs)):
                obs_i = buffer.mdo_obs[i]
                if obs_i.numel() == 0:
                    continue
                action_i = torch.tensor(buffer.mdo_actions[i], dtype=torch.long)
                old_log_prob = buffer.mdo_log_probs[i]
                num_vnfs_i = buffer.mdo_num_vnfs[i]
                if num_vnfs_i == 0 or i >= len(buffer.mdo_tier_masks):
                    continue
                tier_mask_i = buffer.mdo_tier_masks[i]
                adv_i = float(advantages[i]) if i < len(advantages) else 0.0

                new_log_probs, new_entropy, new_logits = mdo_policy.evaluate_actions(
                    obs_i, tier_mask_i, action_i, num_vnfs_i,
                )

                ratio = torch.exp(new_log_probs.sum() - old_log_prob.sum())
                unclipped = ratio * adv_i
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_i
                step_loss = -torch.min(unclipped, clipped)
                entropy_bonus = cfg.entropy_coef * new_entropy

                # KL prior term (only if β > 0)
                kl_term = torch.tensor(0.0)
                beta_t = beta_schedule(
                    total_steps, total_decay_steps,
                    beta_start=kl_beta_start, beta_end=kl_beta_end,
                )
                if beta_t > 0 and i < len(suggested_domains_list):
                    suggested = suggested_domains_list[i]
                    if suggested and len(suggested) == num_vnfs_i:
                        prior_logits = build_prior_logits(
                            suggested, num_domains, tier_mask_i,
                        )
                        kl_term = analytical_kl(
                            new_logits[:num_vnfs_i],
                            prior_logits[:num_vnfs_i],
                            tier_mask_i[:num_vnfs_i] if tier_mask_i.dim() == 2 else None,
                        )

                epoch_loss = epoch_loss + step_loss - entropy_bonus + beta_t * kl_term
                epoch_count += 1

                mdo_policy_loss += step_loss.item()
                mdo_entropy_sum += new_entropy.item()
                kl_sum += kl_term.item()
                mdo_steps += 1
                if ratio.item() < (1 - cfg.clip_eps) or ratio.item() > (1 + cfg.clip_eps):
                    mdo_clips += 1

                total_steps += 1

            if epoch_count > 0:
                mean_loss = epoch_loss / epoch_count
                mdo_optimizer.zero_grad()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(mdo_policy.parameters(), 0.5)
                mdo_optimizer.step()

        # Diagnostics
        param_motion = 0.0
        for n, p in mdo_policy.named_parameters():
            param_motion += (p - initial_params[n]).abs().sum().item()

        # Violation breakdown
        violation_types = Counter()
        for r in ep.mdo_results:
            if not r.admitted:
                for a in r.retry_history.attempts:
                    if a.violation is not None:
                        if a.violation.actor_infeasible:
                            violation_types["actor_infeasible"] += 1
                        if a.violation.c5b_violated:
                            violation_types["c5b"] += 1
                        if a.violation.c7_violated:
                            violation_types["c7"] += 1
                        if a.violation.c9_violated:
                            violation_types["c9"] += 1
                        if a.violation.cross_domain_infeasible:
                            violation_types["cross_domain"] += 1

        # Partition distribution
        domain_counts = Counter()
        for r in ep.mdo_results:
            if r.retry_history.attempts:
                last_attempt = r.retry_history.attempts[-1]
                for d in last_attempt.partition:
                    domain_counts[d] += 1
        total_vnfs = sum(domain_counts.values())
        dist_str = " ".join(
            f"d{d}={c}({100*c/max(total_vnfs,1):.0f}%)"
            for d, c in sorted(domain_counts.items())
        )

        cross_admitted = sum(1 for r in ep.mdo_results if r.admitted and r.cross_domain_routes)
        beta_t = beta_schedule(total_steps, total_decay_steps, kl_beta_start, kl_beta_end)
        elapsed = time.time() - t0
        viol_str = " ".join(f"{k}={v}" for k, v in sorted(violation_types.items()))

        logger.info(
            "[%s] R%d: admit=%d/%d (%.1f%%) cross=%d  clips=%d/%d  "
            "ent=%.3f  β=%.3f  kl=%.4f  motion=%.4f  %.1fs",
            name, rnd + 1, admitted, total, rate, cross_admitted,
            mdo_clips, mdo_steps,
            mdo_entropy_sum / max(mdo_steps, 1),
            beta_t, kl_sum / max(mdo_steps, 1),
            param_motion, elapsed,
        )
        logger.info("  violations: %s  partition: %s", viol_str, dist_str)

        results.append({
            "round": rnd + 1,
            "admitted": admitted,
            "total": total,
            "rate": rate,
            "cross_domain": cross_admitted,
            "clips": mdo_clips,
            "entropy": mdo_entropy_sum / max(mdo_steps, 1),
            "beta": beta_t,
            "kl": kl_sum / max(mdo_steps, 1),
            "param_motion": param_motion,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-end", type=float, default=0.01)
    args = parser.parse_args()

    # Condition 1: β0 (no prior)
    beta0_results = run_condition(
        "β0", args.seed, args.rounds,
        kl_beta_start=0.0, kl_beta_end=0.0, lr=args.lr,
    )

    # Condition 2: annealed β
    anneal_results = run_condition(
        "anneal", args.seed, args.rounds,
        kl_beta_start=args.beta_start, kl_beta_end=args.beta_end, lr=args.lr,
    )

    # Summary comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY: anneal vs β0")
    logger.info("=" * 70)

    def _avg(results, key, start=0, end=None):
        subset = results[start:end] if end else results[start:]
        if not subset:
            return 0.0
        return sum(r[key] for r in subset) / len(subset)

    for label, window_start in [("all rounds", 0), ("last 10", -10), ("last 5", -5)]:
        b0_rate = _avg(beta0_results, "rate", window_start)
        an_rate = _avg(anneal_results, "rate", window_start)
        delta = an_rate - b0_rate
        logger.info(
            "  %s: β0=%.1f%%  anneal=%.1f%%  Δ=%+.1f pp",
            label.ljust(12), b0_rate, an_rate, delta,
        )

    b0_peak = max(r["rate"] for r in beta0_results) if beta0_results else 0
    an_peak = max(r["rate"] for r in anneal_results) if anneal_results else 0
    logger.info("  peak: β0=%.1f%%  anneal=%.1f%%  Δ=%+.1f pp", b0_peak, an_peak, an_peak - b0_peak)

    verdict = "PRIOR HELPS" if _avg(anneal_results, "rate", -10) > _avg(beta0_results, "rate", -10) + 1.0 else "NO SIGNAL"
    logger.info("")
    logger.info("  VERDICT: %s", verdict)


if __name__ == "__main__":
    main()
