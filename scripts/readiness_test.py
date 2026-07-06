#!/usr/bin/env python3
"""Readiness test: does the MDO climb off the random-partition floor?

One seed, β0, sample mode, 5 training rounds. Watch admission rate.
If it moves off ~13.7%, the system learns and the full ablation is go.

Usage:
    python scripts/readiness_test.py --seed 0 --rounds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import compute_cost_greedy, _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.observation import build_mdo_observation, observation_to_tensor
from orion.mdo.policy import MDOPolicy, DirectJointPolicy
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.reward import RewardWeights
from orion.sim.rollout_buffer import DomainActorTransition
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.critic import CentralisedCritic
from orion.training.gae import compute_gae
from orion.training.global_state import probe_global_state_dim
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("readiness")

NUM_ARRIVALS = 200
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
HIDDEN_DIM = 64


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--freeze-actors", action="store_true",
                        help="Freeze domain actors — train only MDO + critic")
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    freeze_actors = args.freeze_actors
    logger.info("=" * 60)
    logger.info("READINESS TEST: β0, sample mode, %d rounds%s",
                args.rounds, " [ACTORS FROZEN]" if freeze_actors else "")
    logger.info("=" * 60)

    # Build substrate and probe dimensions
    substrate = build_substrate(seed)
    delays = build_delays(substrate)

    max_vnfs = 10
    dummy_plan = PlanSummary(
        vnf_ids=["v0", "v1"], required_tiers=[InfrastructureTier.MEC] * 2,
        suggested_domains=[0, 1], cpu_demands=[1.0] * 2, ram_demands=[1.0] * 2,
        vcrs=[1.0] * 2, bw_demands=[10.0],
    )
    obs = build_mdo_observation(substrate, dummy_plan)
    obs_tensor = observation_to_tensor(obs, max_vnfs=max_vnfs)
    obs_dim = obs_tensor.shape[0]
    num_domains = 3

    # Create MDO policy (the thing under test)
    mdo_policy = MDOPolicy(
        obs_dim=obs_dim, num_domains=num_domains,
        max_vnfs=max_vnfs, hidden_dim=128, num_layers=2,
    )

    # Domain actors (untrained, just placing)
    actors = {}
    for d in range(num_domains):
        torch.manual_seed(seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))

    # MDO coordinator with real policy
    coord = MDOCoordinator(mdo_policy, actors, MDOConfig(
        n_part=3, mu=1.0, alpha=0.1, xi=0.1, eta=0.1,
    ))

    # Critic — uses MDO obs as input (per-arrival substrate state)
    critic = CentralisedCritic(input_dim=obs_dim, hidden_dim=128, num_layers=2)

    # Optimizers
    mdo_optimizer = torch.optim.Adam(mdo_policy.parameters(), lr=3e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    domain_optimizers = {}
    for d, actor in actors.items():
        domain_optimizers[d] = torch.optim.Adam(actor.policy.parameters(), lr=3e-4)

    cfg = MAPPOConfig(
        kl_beta_initial=0.0, kl_beta_final=0.0,  # β0: no prior
        update_epochs=4, clip_eps=0.2, entropy_coef=0.01,
        gamma=0.99, gae_lambda=0.95,
    )

    # Snapshot initial MDO policy params for parameter-motion check
    initial_params = {n: p.clone() for n, p in mdo_policy.named_parameters()}
    initial_actor_params = {}
    for d, actor in actors.items():
        for n, p in actor.policy.named_parameters():
            initial_actor_params[f"actor_{d}_{n}"] = p.clone()

    logger.info("obs_dim=%d (critic shares obs input), num_domains=%d", obs_dim, num_domains)

    # Fixed eval stream for per-round deterministic snapshots
    snap_substrate = build_substrate(seed)
    snap_delays = build_delays(snap_substrate)
    snap_rng = np.random.default_rng(seed + 8888)
    snap_ap = ArrivalProcess(snap_substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, snap_rng)
    snap_ap.generate()
    snap_events = list(snap_ap.events)
    det_curve = []

    for rnd in range(args.rounds):
        t0 = time.time()

        # Fresh substrate + arrivals each round
        substrate = build_substrate(seed)
        rng = np.random.default_rng(seed + rnd * 1_000_000 + 1)
        ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
        ap.generate()

        runner = EpisodeRunner(
            substrate, ap, coord, delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        runner.reset()

        # Collect episode in SAMPLE mode — MDO policy chooses partitions
        ep = runner.run_episode(mdo_mode="sample")

        admitted = ep.stats.admitted
        total = ep.stats.total_arrivals
        hard_fires = ep.stats.hard_penalty_fires
        rate = 100 * admitted / total if total > 0 else 0

        # Build PPO buffer from episode — per-transition critic values
        buffer = PPORolloutBuffer()
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
        for domain_id, transitions in ep.rollout.domain_actor.items():
            for t in transitions:
                buffer.append_domain_actor(t)

        if len(buffer) == 0:
            logger.info("Round %d: empty buffer, skipping", rnd)
            continue

        # Contextual bandit: each arrival is independent.
        # Return = own reward, no cross-arrival discounting.
        # Advantage = R - V(s) for each transition independently.
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

        # --- Critic update ---
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

        critic_baseline = sum(buffer.critic_values) / len(buffer.critic_values) if buffer.critic_values else 0.0

        # --- MDO policy PPO update (batched per epoch) ---
        mdo_policy_loss = 0.0
        mdo_entropy_sum = 0.0
        mdo_steps = 0
        mdo_clips = 0

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

                epoch_loss = epoch_loss + step_loss - entropy_bonus
                epoch_count += 1

                mdo_policy_loss += step_loss.item()
                mdo_entropy_sum += new_entropy.item()
                mdo_steps += 1
                if ratio.item() < (1 - cfg.clip_eps) or ratio.item() > (1 + cfg.clip_eps):
                    mdo_clips += 1

            if epoch_count > 0:
                mean_loss = epoch_loss / epoch_count
                mdo_optimizer.zero_grad()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(mdo_policy.parameters(), 0.5)
                mdo_optimizer.step()

        # --- Domain actor PPO update (batched per epoch per domain) ---
        # Per-outcome advantage: normalize feasible and infeasible separately
        # to prevent advantage collapse from class imbalance (94% infeasible).
        from torch.distributions import Categorical

        actor_loss_total = 0.0
        actor_steps = 0
        if not freeze_actors:
            for domain_id, transitions in buffer.domain_actor.items():
                if domain_id not in domain_optimizers:
                    continue
                actor = actors[domain_id]
                optimizer = domain_optimizers[domain_id]

                for _epoch in range(cfg.update_epochs):
                    epoch_actor_loss = torch.tensor(0.0)
                    epoch_actor_count = 0

                    for t in transitions:
                        if not t.steps:
                            continue
                        advantage = t.terminal_reward - critic_baseline

                        for step in t.steps:
                            new_logits = actor.policy._encode_and_score(
                                step.graph_data, step.vnf_context, step.action_mask,
                            )
                            dist = Categorical(logits=new_logits)
                            n_nodes = step.action_mask.size(0)
                            action_tensor = torch.tensor(
                                n_nodes if step.action_idx == DomainPolicy.NULL_ACTION else step.action_idx
                            )
                            new_lp = dist.log_prob(action_tensor)
                            ratio = torch.exp(new_lp - step.log_prob)
                            unclipped = ratio * advantage
                            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantage
                            step_loss = -torch.min(unclipped, clipped)
                            entropy_bonus = cfg.entropy_coef * dist.entropy()

                            epoch_actor_loss = epoch_actor_loss + step_loss - entropy_bonus
                            epoch_actor_count += 1
                            actor_loss_total += step_loss.item()
                            actor_steps += 1

                    if epoch_actor_count > 0:
                        mean_actor_loss = epoch_actor_loss / epoch_actor_count
                        optimizer.zero_grad()
                        mean_actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.policy.parameters(), 0.5)
                        optimizer.step()

        # --- Actor infeasibility root cause: empty mask vs policy NULL ---
        from orion.actors.policy import DomainPolicy as DP
        null_with_options = 0
        null_forced = 0
        routing_fail = 0
        for domain_id, transitions in buffer.domain_actor.items():
            for t in transitions:
                if t.accepted:
                    continue
                if not t.steps:
                    routing_fail += 1
                    continue
                last_step = t.steps[-1]
                mask_any = last_step.action_mask.any().item()
                is_null = last_step.action_idx == DP.NULL_ACTION
                if is_null and mask_any:
                    null_with_options += 1
                elif is_null and not mask_any:
                    null_forced += 1
                elif not is_null:
                    routing_fail += 1
        logger.info(
            "  infeasibility: null_forced(mask empty)=%d  null_chosen(had options)=%d  routing_fail=%d",
            null_forced, null_with_options, routing_fail,
        )

        # --- Actor learning diagnostics: advantage split by outcome ---
        actor_adv_feasible = {d: [] for d in range(num_domains)}
        actor_adv_infeasible = {d: [] for d in range(num_domains)}
        actor_param_motion = {}
        for domain_id, transitions in buffer.domain_actor.items():
            if domain_id >= num_domains:
                continue
            for t in transitions:
                if not t.steps:
                    continue
                adv = t.terminal_reward - critic_baseline
                if t.accepted:
                    actor_adv_feasible[domain_id].append(adv)
                else:
                    actor_adv_infeasible[domain_id].append(adv)
        if not freeze_actors:
            for d, actor in actors.items():
                motion = 0.0
                for n, p in actor.policy.named_parameters():
                    key = f"actor_{d}_{n}"
                    if key not in initial_actor_params:
                        motion += p.abs().sum().item()
                    else:
                        motion += (p - initial_actor_params[key]).abs().sum().item()
                actor_param_motion[d] = motion
        for d in range(num_domains):
            feas = actor_adv_feasible[d]
            infeas = actor_adv_infeasible[d]
            f_str = f"mean={sum(feas)/len(feas):.4f} std={torch.tensor(feas).std():.4f}" if feas else "n=0"
            i_str = f"mean={sum(infeas)/len(infeas):.4f} std={torch.tensor(infeas).std():.4f}" if infeas else "n=0"
            motion_str = f"motion={actor_param_motion.get(d, 0):.4f}" if not freeze_actors else ""
            logger.info(
                "  actor[%d]: feasible=%d infeasible=%d  %s",
                d, len(feas), len(infeas), motion_str,
            )
            logger.info(
                "    adv(feasible): %s  adv(infeasible): %s  baseline=%.4f",
                f_str, i_str, critic_baseline,
            )

        # --- Advantage/value diagnostics ---
        adv_admit, adv_reject, val_admit, val_reject = [], [], [], []
        ret_admit, ret_reject = [], []
        for i in range(len(buffer.rewards)):
            r = buffer.rewards[i]
            v = buffer.critic_values[i]
            a = float(advantages[i]) if i < len(advantages) else 0.0
            ret_i = float(returns[i]) if i < len(returns) else 0.0
            if r > 0:
                adv_admit.append(a)
                val_admit.append(v)
                ret_admit.append(ret_i)
            else:
                adv_reject.append(a)
                val_reject.append(v)
                ret_reject.append(ret_i)

        def _stats(xs):
            if not xs:
                return "n=0"
            t = torch.tensor(xs)
            return f"n={len(xs)} mean={t.mean():.3f} std={t.std():.3f} min={t.min():.3f} max={t.max():.3f}"

        # Parameter motion check
        param_motion = 0.0
        for n, p in mdo_policy.named_parameters():
            param_motion += (p - initial_params[n]).abs().sum().item()

        elapsed = time.time() - t0
        cross_admitted = sum(1 for r in ep.mdo_results if r.admitted and r.cross_domain_routes)

        logger.info(
            "Round %d/%d: admitted=%d/%d (%.1f%%) gated=%d cross_domain=%d "
            "mdo_loss=%.4f mdo_ent=%.3f mdo_clips=%d/%d "
            "critic_loss=%.4f actor_loss=%.4f "
            "param_motion=%.4f  %.1fs",
            rnd + 1, args.rounds, admitted, total, rate, hard_fires, cross_admitted,
            mdo_policy_loss / max(mdo_steps, 1),
            mdo_entropy_sum / max(mdo_steps, 1),
            mdo_clips, mdo_steps,
            critic_loss_val / cfg.update_epochs,
            actor_loss_total / max(actor_steps, 1),
            param_motion, elapsed,
        )
        logger.info("  value_loss=%.1f  V(admit): %s", critic_loss_val / cfg.update_epochs, _stats(val_admit))
        logger.info("  V(reject): %s", _stats(val_reject))
        logger.info("  ret(admit): %s  ret(reject): %s", _stats(ret_admit), _stats(ret_reject))
        logger.info("  adv(admit): %s", _stats(adv_admit))
        logger.info("  adv(reject): %s", _stats(adv_reject))

        # Cost distribution check: is the MDO cherry-picking cheap slices?
        admit_costs = [r.total_cost for r in ep.mdo_results if r.admitted]
        admit_rewards_list = [r.reward.total for r in ep.mdo_results if r.admitted]
        reject_feasible_costs = []
        for r in ep.mdo_results:
            if not r.admitted and r.retry_history.attempts:
                for a in r.retry_history.attempts:
                    if a.total_cost > 0:
                        reject_feasible_costs.append(a.total_cost)
                        break
        logger.info("  cost(admitted): %s", _stats(admit_costs))
        logger.info("  cost(rejected w/ nonzero cost): %s", _stats(reject_feasible_costs))
        logger.info("  reward(admitted): %s", _stats(admit_rewards_list))

        # Reject reason breakdown
        from orion.mdo.types import RejectReason
        from collections import Counter
        reject_reasons = Counter()
        violation_types = Counter()
        structural_rejects = ep.stats.rejected_structural
        for r in ep.mdo_results:
            if not r.admitted:
                if r.reject_reason is not None:
                    reject_reasons[r.reject_reason.name] += 1
                elif r.action == 0:
                    reject_reasons["VERIFIER_GATED"] += 1
                else:
                    reject_reasons["UNKNOWN"] += 1
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
        rej_str = " ".join(f"{k}={v}" for k, v in sorted(reject_reasons.items()))
        viol_str = " ".join(f"{k}={v}" for k, v in sorted(violation_types.items()))
        logger.info("  reject_reasons: %s  structural=%d", rej_str, structural_rejects)
        logger.info("  violation_types: %s", viol_str)

        # Partition distribution: how many VNFs assigned to each domain
        domain_counts = Counter()
        for r in ep.mdo_results:
            if r.retry_history.attempts:
                last_attempt = r.retry_history.attempts[-1]
                for d in last_attempt.partition:
                    domain_counts[d] += 1
        total_vnfs = sum(domain_counts.values())
        dist_str = " ".join(f"d{d}={c}/{total_vnfs}({100*c/max(total_vnfs,1):.0f}%)" for d, c in sorted(domain_counts.items()))
        logger.info("  partition_dist: %s", dist_str)

        # Per-round ratio diagnostic: how far do ratios get from 1.0?
        ratio_devs = []
        for i in range(len(buffer.mdo_obs)):
            obs_i = buffer.mdo_obs[i]
            if obs_i.numel() == 0:
                continue
            action_i = torch.tensor(buffer.mdo_actions[i], dtype=torch.long)
            old_lp = buffer.mdo_log_probs[i]
            num_vnfs_i = buffer.mdo_num_vnfs[i]
            if num_vnfs_i == 0 or i >= len(buffer.mdo_tier_masks):
                continue
            tm_i = buffer.mdo_tier_masks[i]
            with torch.no_grad():
                new_lp, _, _ = mdo_policy.evaluate_actions(obs_i, tm_i, action_i, num_vnfs_i)
            r = torch.exp(new_lp.sum() - old_lp.sum()).item()
            ratio_devs.append(abs(r - 1.0))
        if ratio_devs:
            rt = torch.tensor(ratio_devs)
            logger.info("  ratio |r-1|: mean=%.4f max=%.4f median=%.4f", rt.mean(), rt.max(), rt.median())

        # Per-round deterministic snapshot on fixed eval stream
        with torch.no_grad():
            snap_sub = build_substrate(seed)
            snap_d = build_delays(snap_sub)
            snap_ap_run = ArrivalProcess(snap_sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, snap_rng)
            snap_ap_run.events = list(snap_events)
            snap_ap_run._event_idx = 0
            snap_runner = EpisodeRunner(
                snap_sub, snap_ap_run, coord, snap_d,
                plan_builder=_greedy_plan_builder,
                reward_weights=RewardWeights(lambda_viol=10.0),
            )
            snap_ep = snap_runner.run_episode(mdo_mode="deterministic")
            snap_admit = snap_ep.stats.admitted
            snap_total = snap_ep.stats.total_arrivals
            snap_rate = 100 * snap_admit / max(snap_total, 1)
            snap_cost = sum(r.total_cost for r in snap_ep.mdo_results if r.admitted)
            snap_departs = snap_ep.stats.departures
            snap_entropy = mdo_entropy_sum / max(mdo_steps, 1) if mdo_steps > 0 else 0.0
            det_curve.append((rnd + 1, snap_rate, snap_cost, snap_entropy, snap_departs))
            logger.info(
                "  [DET SNAPSHOT] round=%d admit=%.1f%% (%d/%d) cost=%.1f departs=%d entropy=%.3f",
                rnd + 1, snap_rate, snap_admit, snap_total, snap_cost, snap_departs, snap_entropy,
            )

    # Final parameter motion
    total_motion = 0.0
    for n, p in mdo_policy.named_parameters():
        total_motion += (p - initial_params[n]).abs().sum().item()

    logger.info("")
    logger.info("=" * 60)
    logger.info("READINESS VERDICT")
    logger.info("=" * 60)
    logger.info("  MDO parameter motion: %.6f", total_motion)
    if total_motion > 0.01:
        logger.info("  MDO is LEARNING (params moved)")
    else:
        logger.info("  MDO is FROZEN (params didn't move)")

    # ── Paired baseline: ORION vs greedy on K identical streams ──────────
    K_EVAL = 5
    logger.info("")
    logger.info("=" * 60)
    logger.info("PAIRED EVALUATION: %d identical streams", K_EVAL)
    logger.info("=" * 60)

    orion_rates = []
    greedy_rates = []
    orion_costs = []
    greedy_costs = []
    orion_gated = []
    greedy_gated = []

    for k in range(K_EVAL):
        eval_seed = seed + 9000 + k
        eval_substrate = build_substrate(seed)
        eval_delays = build_delays(eval_substrate)
        eval_rng = np.random.default_rng(eval_seed)
        eval_ap = ArrivalProcess(eval_substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap.generate()
        eval_events = list(eval_ap.events)

        # Run ORION (deterministic — no exploration noise)
        eval_substrate_orion = build_substrate(seed)
        eval_ap_orion = ArrivalProcess(eval_substrate_orion, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap_orion.events = list(eval_events)
        eval_ap_orion._event_idx = 0

        orion_runner = EpisodeRunner(
            eval_substrate_orion, eval_ap_orion, coord, eval_delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        orion_ep = orion_runner.run_episode(mdo_mode="deterministic")

        # Run greedy on identical substrate + stream
        eval_substrate_greedy = build_substrate(seed)
        eval_ap_greedy = ArrivalProcess(eval_substrate_greedy, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap_greedy.events = list(eval_events)
        eval_ap_greedy._event_idx = 0

        greedy_coord = MDOCoordinator(None, actors, MDOConfig(
            n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1,
        ))
        greedy_runner = EpisodeRunner(
            eval_substrate_greedy, eval_ap_greedy, greedy_coord, eval_delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        greedy_ep = greedy_runner.run_episode(mdo_mode="follow_prior")

        o_rate = 100 * orion_ep.stats.admitted / max(orion_ep.stats.total_arrivals, 1)
        g_rate = 100 * greedy_ep.stats.admitted / max(greedy_ep.stats.total_arrivals, 1)
        o_cost = sum(r.total_cost for r in orion_ep.mdo_results if r.admitted)
        g_cost = sum(r.total_cost for r in greedy_ep.mdo_results if r.admitted)
        o_mean_cost = o_cost / max(orion_ep.stats.admitted, 1)
        g_mean_cost = g_cost / max(greedy_ep.stats.admitted, 1)

        orion_rates.append(o_rate)
        greedy_rates.append(g_rate)
        orion_costs.append(o_mean_cost)
        greedy_costs.append(g_mean_cost)
        orion_gated.append(orion_ep.stats.hard_penalty_fires)
        greedy_gated.append(greedy_ep.stats.hard_penalty_fires)

        logger.info(
            "  stream %d: ORION=%.1f%% cost/slice=%.1f gated=%d  "
            "greedy=%.1f%% cost/slice=%.1f gated=%d  "
            "admit_diff=%+.1f%% cost_diff=%+.1f",
            k,
            o_rate, o_mean_cost, orion_ep.stats.hard_penalty_fires,
            g_rate, g_mean_cost, greedy_ep.stats.hard_penalty_fires,
            o_rate - g_rate, o_mean_cost - g_mean_cost,
        )

    orion_mean = sum(orion_rates) / K_EVAL
    greedy_mean = sum(greedy_rates) / K_EVAL
    orion_mean_cost = sum(orion_costs) / K_EVAL
    greedy_mean_cost = sum(greedy_costs) / K_EVAL
    diffs = [o - g for o, g in zip(orion_rates, greedy_rates)]
    diff_mean = sum(diffs) / K_EVAL
    positive = sum(1 for d in diffs if d > 0)

    logger.info("")
    logger.info(
        "  ORION  admit=%.1f%% cost/slice=%.1f  |  greedy  admit=%.1f%% cost/slice=%.1f",
        orion_mean, orion_mean_cost, greedy_mean, greedy_mean_cost,
    )
    logger.info(
        "  admit diff=%+.1f%%  cost diff=%+.1f  positive=%d/%d",
        diff_mean, orion_mean_cost - greedy_mean_cost, positive, K_EVAL,
    )

    # ── Sequential-argmax test: capacity-aware decoding ───────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SEQUENTIAL ARGMAX: same weights, capacity-aware decoding")
    logger.info("=" * 60)

    seq_rates = []
    seq_costs = []
    for k in range(K_EVAL):
        eval_seed = seed + 9000 + k
        eval_substrate = build_substrate(seed)
        eval_delays = build_delays(eval_substrate)
        eval_rng = np.random.default_rng(eval_seed)
        eval_ap = ArrivalProcess(eval_substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap.generate()
        eval_events = list(eval_ap.events)

        eval_substrate_seq = build_substrate(seed)
        eval_ap_seq = ArrivalProcess(eval_substrate_seq, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap_seq.events = list(eval_events)
        eval_ap_seq._event_idx = 0
        seq_runner = EpisodeRunner(
            eval_substrate_seq, eval_ap_seq, coord, eval_delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        seq_ep = seq_runner.run_episode(mdo_mode="sequential_argmax")

        s_rate = 100 * seq_ep.stats.admitted / max(seq_ep.stats.total_arrivals, 1)
        s_cost = sum(r.total_cost for r in seq_ep.mdo_results if r.admitted)
        s_mean_cost = s_cost / max(seq_ep.stats.admitted, 1)
        seq_rates.append(s_rate)
        seq_costs.append(s_mean_cost)
        logger.info(
            "  stream %d: SEQ=%.1f%% cost/slice=%.1f gated=%d",
            k, s_rate, s_mean_cost, seq_ep.stats.hard_penalty_fires,
        )

    seq_mean = sum(seq_rates) / K_EVAL
    seq_mean_cost = sum(seq_costs) / K_EVAL
    logger.info("")
    logger.info(
        "  SEQ_ARGMAX admit=%.1f%% cost=%.1f  |  IND_ARGMAX admit=%.1f%% cost=%.1f  |  greedy admit=%.1f%% cost=%.1f",
        seq_mean, seq_mean_cost, orion_mean, orion_mean_cost, greedy_mean, greedy_mean_cost,
    )
    logger.info("  Gap recovered: %.1f%% of the %.1f%% argmax-to-sample gap",
                seq_mean - orion_mean, 56.0 - orion_mean)

    # ── Masked-random baseline: uniform over tier mask, no policy ──────
    logger.info("")
    logger.info("=" * 60)
    logger.info("MASKED-RANDOM BASELINE: uniform over feasible domains")
    logger.info("=" * 60)

    rand_rates = []
    rand_costs = []
    for k in range(K_EVAL):
        eval_seed = seed + 9000 + k
        eval_substrate = build_substrate(seed)
        eval_delays = build_delays(eval_substrate)
        eval_rng = np.random.default_rng(eval_seed)
        eval_ap = ArrivalProcess(eval_substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap.generate()
        eval_events = list(eval_ap.events)

        eval_substrate_r = build_substrate(seed)
        eval_ap_r = ArrivalProcess(eval_substrate_r, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, eval_rng)
        eval_ap_r.events = list(eval_events)
        eval_ap_r._event_idx = 0
        rand_runner = EpisodeRunner(
            eval_substrate_r, eval_ap_r, coord, eval_delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        rand_ep = rand_runner.run_episode(mdo_mode="random")

        r_rate = 100 * rand_ep.stats.admitted / max(rand_ep.stats.total_arrivals, 1)
        r_cost = sum(r.total_cost for r in rand_ep.mdo_results if r.admitted)
        r_mean_cost = r_cost / max(rand_ep.stats.admitted, 1)
        rand_rates.append(r_rate)
        rand_costs.append(r_mean_cost)
        logger.info(
            "  stream %d: RANDOM=%.1f%% cost/slice=%.1f gated=%d",
            k, r_rate, r_mean_cost, rand_ep.stats.hard_penalty_fires,
        )

    rand_mean = sum(rand_rates) / K_EVAL
    rand_mean_cost = sum(rand_costs) / K_EVAL
    logger.info("")
    logger.info(
        "  RANDOM admit=%.1f%% cost=%.1f  |  LEARNED_SAMPLE≈57%%  |  IND_ARGMAX admit=%.1f%% cost=%.1f  |  greedy admit=%.1f%% cost=%.1f",
        rand_mean, rand_mean_cost, orion_mean, orion_mean_cost, greedy_mean, greedy_mean_cost,
    )
    logger.info("  If RANDOM ≈ 57%%: learned sample = random, MDO learned nothing over feasibility")
    logger.info("  If RANDOM < 57%%: MDO learned a useful soft distribution, 'learned nothing' is wrong")

    # ── Direct-joint policy: full joint Categorical, no factorization ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("DIRECT-JOINT POLICY: single Categorical over M^K partitions")
    logger.info("=" * 60)

    torch.manual_seed(seed)
    joint_policy = DirectJointPolicy(
        obs_dim=obs_dim, num_domains=num_domains,
        max_chain_length=5, hidden_dim=128, num_layers=2,
    )
    joint_critic = CentralisedCritic(input_dim=obs_dim, hidden_dim=128, num_layers=2)
    joint_mdo_opt = torch.optim.Adam(joint_policy.parameters(), lr=3e-3)
    joint_crit_opt = torch.optim.Adam(joint_critic.parameters(), lr=3e-4)

    joint_cfg = MDOConfig(n_part=3, mu=1.0, alpha=0.1, xi=0.1, eta=0.1)
    joint_coord = MDOCoordinator(joint_policy, actors, joint_cfg)

    for rnd_j in range(args.rounds):
        j_sub = build_substrate(seed)
        j_delays = build_delays(j_sub)
        j_rng = np.random.default_rng(seed + rnd_j * 1_000_000 + 1)
        j_ap = ArrivalProcess(j_sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, j_rng)
        j_ap.generate()
        j_runner = EpisodeRunner(
            j_sub, j_ap, joint_coord, j_delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        j_runner.reset()
        j_ep = j_runner.run_episode(mdo_mode="sample")

        j_buf = PPORolloutBuffer()
        with torch.no_grad():
            for t in j_ep.rollout.mdo:
                obs_fc = t.obs if t.obs.numel() > 0 else torch.zeros(obs_dim)
                cv = float(joint_critic(obs_fc.unsqueeze(0)).item())
                j_buf.append_mdo(
                    mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                    entropy=t.entropy, aux_value=t.value_estimate,
                    global_state=obs_fc, critic_value=cv,
                    reward=t.terminal_reward, done=t.committed,
                    tier_mask=t.tier_mask, num_vnfs=t.num_vnfs,
                )
        if len(j_buf) == 0:
            continue

        j_rewards = j_buf.reward_tensor()
        with torch.no_grad():
            j_vals = torch.tensor([
                float(joint_critic(o.unsqueeze(0)).item()) if o.numel() > 0 else 0.0
                for o in j_buf.mdo_obs
            ], dtype=torch.float32)
        j_advs = j_rewards - j_vals
        j_rets = j_rewards.clone()
        j_buf.set_gae(j_advs, j_rets)

        for _ in range(cfg.update_epochs):
            gs = torch.stack(j_buf.global_states)
            nv = joint_critic(gs).squeeze(-1)
            old_v = torch.tensor(j_buf.critic_values, dtype=torch.float32)
            vc = old_v + torch.clamp(nv - old_v, -cfg.clip_eps, cfg.clip_eps)
            vl = 0.5 * torch.max((nv - j_rets) ** 2, (vc - j_rets) ** 2).mean()
            joint_crit_opt.zero_grad()
            (0.5 * vl).backward()
            torch.nn.utils.clip_grad_norm_(joint_critic.parameters(), 0.5)
            joint_crit_opt.step()

        for _ in range(cfg.update_epochs):
            el = torch.tensor(0.0)
            ec = 0
            for i in range(len(j_buf.mdo_obs)):
                oi = j_buf.mdo_obs[i]
                if oi.numel() == 0:
                    continue
                ai = torch.tensor(j_buf.mdo_actions[i], dtype=torch.long)
                olp = j_buf.mdo_log_probs[i]
                nvi = j_buf.mdo_num_vnfs[i]
                if nvi == 0 or i >= len(j_buf.mdo_tier_masks):
                    continue
                tmi = j_buf.mdo_tier_masks[i]
                advi = float(j_advs[i]) if i < len(j_advs) else 0.0
                nlp, ne, _ = joint_policy.evaluate_actions(oi, tmi, ai, nvi)
                r = torch.exp(nlp.sum() - olp.sum())
                u = r * advi
                c = torch.clamp(r, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advi
                el = el + (-torch.min(u, c)) - cfg.entropy_coef * ne
                ec += 1
            if ec > 0:
                joint_mdo_opt.zero_grad()
                (el / ec).backward()
                torch.nn.utils.clip_grad_norm_(joint_policy.parameters(), 0.5)
                joint_mdo_opt.step()

        j_rate = 100 * j_ep.stats.admitted / max(j_ep.stats.total_arrivals, 1)
        logger.info("  joint round %d: sample admit=%.1f%% gated=%d",
                     rnd_j + 1, j_rate, j_ep.stats.hard_penalty_fires)

    # Eval direct-joint policy on same K streams
    logger.info("  --- direct-joint paired eval ---")
    joint_rates = []
    joint_costs = []
    joint_ents = []
    for k in range(K_EVAL):
        eval_seed = seed + 9000 + k
        es = build_substrate(seed)
        ed = build_delays(es)
        er = np.random.default_rng(eval_seed)
        ea = ArrivalProcess(es, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, er)
        ea.generate()
        ee = list(ea.events)

        es2 = build_substrate(seed)
        ea2 = ArrivalProcess(es2, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, er)
        ea2.events = list(ee)
        ea2._event_idx = 0
        jr = EpisodeRunner(
            es2, ea2, joint_coord, ed,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        je = jr.run_episode(mdo_mode="deterministic")
        jr2 = 100 * je.stats.admitted / max(je.stats.total_arrivals, 1)
        jc = sum(r.total_cost for r in je.mdo_results if r.admitted)
        jmc = jc / max(je.stats.admitted, 1)
        joint_rates.append(jr2)
        joint_costs.append(jmc)
        logger.info(
            "  stream %d: JOINT=%.1f%% cost/slice=%.1f gated=%d",
            k, jr2, jmc, je.stats.hard_penalty_fires,
        )

    joint_mean = sum(joint_rates) / K_EVAL
    joint_mean_cost = sum(joint_costs) / K_EVAL
    logger.info("")
    logger.info(
        "  JOINT admit=%.1f%% cost=%.1f  |  FACTORED admit=%.1f%% cost=%.1f  |  greedy admit=%.1f%% cost=%.1f",
        joint_mean, joint_mean_cost, orion_mean, orion_mean_cost, greedy_mean, greedy_mean_cost,
    )
    logger.info("  If JOINT > greedy: ceiling was representational → autoregressive decoder justified")
    logger.info("  If JOINT ≈ FACTORED: ceiling is credit assignment → fix reward before decoder")

    # ── Learning curve summary ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("DETERMINISTIC LEARNING CURVE (fixed stream)")
    logger.info("=" * 60)
    logger.info("  %5s  %8s  %10s  %8s  %8s", "round", "admit%", "total_cost", "entropy", "departs")
    for rnd_i, rate_i, cost_i, ent_i, dep_i in det_curve:
        logger.info("  %5d  %7.1f%%  %10.1f  %8.3f  %8d", rnd_i, rate_i, cost_i, ent_i, dep_i)

    # ── Admission-pure ablation: train fresh policy with α=0, ξ=0 ────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("ADMISSION-PURE ABLATION: train fresh policy with α=0 ξ=0")
    logger.info("=" * 60)

    torch.manual_seed(seed)
    pure_policy = MDOPolicy(
        obs_dim=obs_dim, num_domains=num_domains,
        max_vnfs=max_vnfs, hidden_dim=128, num_layers=2,
    )
    pure_critic = CentralisedCritic(input_dim=obs_dim, hidden_dim=128, num_layers=2)
    pure_mdo_opt = torch.optim.Adam(pure_policy.parameters(), lr=3e-3)
    pure_crit_opt = torch.optim.Adam(pure_critic.parameters(), lr=3e-4)

    pure_cfg = MDOConfig(n_part=3, mu=1.0, alpha=0.0, xi=0.0, eta=0.0)
    pure_coord = MDOCoordinator(pure_policy, actors, pure_cfg)

    for rnd_p in range(args.rounds):
        p_sub = build_substrate(seed)
        p_delays = build_delays(p_sub)
        p_rng = np.random.default_rng(seed + rnd_p * 1_000_000 + 1)
        p_ap = ArrivalProcess(p_sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, p_rng)
        p_ap.generate()
        p_runner = EpisodeRunner(
            p_sub, p_ap, pure_coord, p_delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        p_runner.reset()
        p_ep = p_runner.run_episode(mdo_mode="sample")

        p_buf = PPORolloutBuffer()
        with torch.no_grad():
            for t in p_ep.rollout.mdo:
                obs_fc = t.obs if t.obs.numel() > 0 else torch.zeros(obs_dim)
                cv = float(pure_critic(obs_fc.unsqueeze(0)).item())
                p_buf.append_mdo(
                    mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                    entropy=t.entropy, aux_value=t.value_estimate,
                    global_state=obs_fc, critic_value=cv,
                    reward=t.terminal_reward, done=t.committed,
                    tier_mask=t.tier_mask, num_vnfs=t.num_vnfs,
                )
        if len(p_buf) == 0:
            continue

        p_rewards = p_buf.reward_tensor()
        with torch.no_grad():
            p_vals = torch.tensor([
                float(pure_critic(o.unsqueeze(0)).item()) if o.numel() > 0 else 0.0
                for o in p_buf.mdo_obs
            ], dtype=torch.float32)
        p_advs = p_rewards - p_vals
        p_rets = p_rewards.clone()
        p_buf.set_gae(p_advs, p_rets)

        # Critic update
        for _ in range(cfg.update_epochs):
            gs = torch.stack(p_buf.global_states)
            nv = pure_critic(gs).squeeze(-1)
            old_v = torch.tensor(p_buf.critic_values, dtype=torch.float32)
            vc = old_v + torch.clamp(nv - old_v, -cfg.clip_eps, cfg.clip_eps)
            vl = 0.5 * torch.max((nv - p_rets) ** 2, (vc - p_rets) ** 2).mean()
            pure_crit_opt.zero_grad()
            (0.5 * vl).backward()
            torch.nn.utils.clip_grad_norm_(pure_critic.parameters(), 0.5)
            pure_crit_opt.step()

        # MDO policy update
        for _ in range(cfg.update_epochs):
            el = torch.tensor(0.0)
            ec = 0
            for i in range(len(p_buf.mdo_obs)):
                oi = p_buf.mdo_obs[i]
                if oi.numel() == 0:
                    continue
                ai = torch.tensor(p_buf.mdo_actions[i], dtype=torch.long)
                olp = p_buf.mdo_log_probs[i]
                nvi = p_buf.mdo_num_vnfs[i]
                if nvi == 0 or i >= len(p_buf.mdo_tier_masks):
                    continue
                tmi = p_buf.mdo_tier_masks[i]
                advi = float(p_advs[i]) if i < len(p_advs) else 0.0
                nlp, ne, _ = pure_policy.evaluate_actions(oi, tmi, ai, nvi)
                r = torch.exp(nlp.sum() - olp.sum())
                u = r * advi
                c = torch.clamp(r, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advi
                el = el + (-torch.min(u, c)) - cfg.entropy_coef * ne
                ec += 1
            if ec > 0:
                pure_mdo_opt.zero_grad()
                (el / ec).backward()
                torch.nn.utils.clip_grad_norm_(pure_policy.parameters(), 0.5)
                pure_mdo_opt.step()

        p_rate = 100 * p_ep.stats.admitted / max(p_ep.stats.total_arrivals, 1)
        logger.info("  ablation round %d: sample admit=%.1f%% gated=%d",
                     rnd_p + 1, p_rate, p_ep.stats.hard_penalty_fires)

    # Eval ablation policy on same K streams
    logger.info("  --- ablation paired eval ---")
    pure_rates = []
    pure_costs = []
    for k in range(K_EVAL):
        eval_seed = seed + 9000 + k
        es = build_substrate(seed)
        ed = build_delays(es)
        er = np.random.default_rng(eval_seed)
        ea = ArrivalProcess(es, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, er)
        ea.generate()
        ee = list(ea.events)

        es2 = build_substrate(seed)
        ea2 = ArrivalProcess(es2, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, er)
        ea2.events = list(ee)
        ea2._event_idx = 0
        pr = EpisodeRunner(
            es2, ea2, pure_coord, ed,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        pe = pr.run_episode(mdo_mode="deterministic")
        pr2 = 100 * pe.stats.admitted / max(pe.stats.total_arrivals, 1)
        pc = sum(r.total_cost for r in pe.mdo_results if r.admitted)
        pmc = pc / max(pe.stats.admitted, 1)
        pure_rates.append(pr2)
        pure_costs.append(pmc)
        logger.info(
            "  stream %d: PURE=%.1f%% cost/slice=%.1f gated=%d",
            k, pr2, pmc, pe.stats.hard_penalty_fires,
        )

    pure_mean = sum(pure_rates) / K_EVAL
    pure_mean_cost = sum(pure_costs) / K_EVAL
    logger.info("")
    logger.info(
        "  PURE(α=0) admit=%.1f%% cost=%.1f  |  ORION(α=0.1) admit=%.1f%% cost=%.1f  |  greedy admit=%.1f%% cost=%.1f",
        pure_mean, pure_mean_cost, orion_mean, orion_mean_cost, greedy_mean, greedy_mean_cost,
    )
    logger.info("  If PURE ≈ greedy: cost terms explain the gap. If PURE < greedy: learning ceiling.")


if __name__ == "__main__":
    main()
