#!/usr/bin/env python3
"""Step A: β > 0 with greedy plan as KL prior, multi-seed.

Tests whether nudging the MDO policy toward the greedy partition via
KL regularisation lifts admission toward greedy levels. Reports mean
AND variance — a good prior should tighten the spread.

Configs:
  ORION_BETA0   — current system, β=0 (baseline from previous eval)
  ORION_PRIOR   — β>0, KL toward greedy-derived prior logits
  RANDOM_FEAS   — random-feasible MDO, no learning (is the selector adding value?)
  GREEDY        — greedy FFD baseline

Usage:
    python scripts/prior_eval.py --rounds 10 --seeds 5 --beta 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.kl_prior import analytical_kl, build_prior_logits
from orion.mdo.observation import build_mdo_observation, observation_to_tensor
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.reward import RewardWeights
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.training.buffer import PPORolloutBuffer
from orion.training.mappo_trainer import CentralisedCritic, MAPPOConfig
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_ARRIVALS = 200
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
K_EVAL = 5


def build_substrate(seed):
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(
            num_domains=3, nodes_per_domain=[8, 10, 12],
            intra_link_density=0.4, inter_domain_links=4,
        ), rng,
    )


def build_delays(sub):
    delays = {}
    for u, v in sub.graph.edges():
        u_dom = sub.graph.nodes[u].get("domain_id", -1)
        v_dom = sub.graph.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            key = (min(u_dom, v_dom), max(u_dom, v_dom))
            delays.setdefault(key, 2.0)
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


def train_policy(seed, rounds, alpha, xi, eta, obs_dim, num_domains, max_vnfs,
                 kl_beta=0.0, prior_temp=1.0):
    """Train one policy from scratch. If kl_beta > 0, add KL toward greedy prior."""
    torch.manual_seed(seed)

    policy = MDOPolicy(
        obs_dim=obs_dim, num_domains=num_domains,
        max_vnfs=max_vnfs, hidden_dim=128, num_layers=2,
    )
    critic = CentralisedCritic(input_dim=obs_dim, hidden_dim=128, num_layers=2)
    mdo_opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    crit_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    actors = {}
    for d in range(num_domains):
        torch.manual_seed(seed + d)
        actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64))
    actor_opts = {d: torch.optim.Adam(a.policy.parameters(), lr=3e-4) for d, a in actors.items()}

    cfg_mdo = MDOConfig(n_part=3, mu=1.0, alpha=alpha, xi=xi, eta=eta)
    coord = MDOCoordinator(policy, actors, cfg_mdo)
    cfg = MAPPOConfig(
        kl_beta_initial=0.0, kl_beta_final=0.0,
        update_epochs=4, clip_eps=0.2, entropy_coef=0.01,
    )

    for rnd in range(rounds):
        sub = build_substrate(seed)
        delays = build_delays(sub)
        rng = np.random.default_rng(seed + rnd * 1_000_000 + 1)
        ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
        ap.generate()
        runner = EpisodeRunner(
            sub, ap, coord, delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        runner.reset()
        ep = runner.run_episode(mdo_mode="sample")

        buf = PPORolloutBuffer()
        suggested_domains_list = []
        with torch.no_grad():
            for t in ep.rollout.mdo:
                obs_fc = t.obs if t.obs.numel() > 0 else torch.zeros(obs_dim)
                cv = float(critic(obs_fc.unsqueeze(0)).item())
                buf.append_mdo(
                    mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                    entropy=t.entropy, aux_value=t.value_estimate,
                    global_state=obs_fc, critic_value=cv,
                    reward=t.terminal_reward, done=t.committed,
                    tier_mask=t.tier_mask, num_vnfs=t.num_vnfs,
                )
                suggested_domains_list.append(t.info.get("suggested_domains", []))
        for domain_id, transitions in ep.rollout.domain_actor.items():
            for t in transitions:
                buf.append_domain_actor(t)
        if len(buf) == 0:
            continue

        rewards = buf.reward_tensor()
        with torch.no_grad():
            vals = torch.tensor([
                float(critic(o.unsqueeze(0)).item()) if o.numel() > 0 else 0.0
                for o in buf.mdo_obs
            ], dtype=torch.float32)
        advs = rewards - vals
        rets = rewards.clone()
        buf.set_gae(advs, rets)

        critic_baseline = float(vals.mean())

        # Critic update
        for _ in range(cfg.update_epochs):
            gs = torch.stack(buf.global_states)
            nv = critic(gs).squeeze(-1)
            old_v = torch.tensor(buf.critic_values, dtype=torch.float32)
            vc = old_v + torch.clamp(nv - old_v, -cfg.clip_eps, cfg.clip_eps)
            vl = 0.5 * torch.max((nv - rets) ** 2, (vc - rets) ** 2).mean()
            crit_opt.zero_grad()
            (0.5 * vl).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            crit_opt.step()

        # MDO policy update (with optional KL prior)
        for _ in range(cfg.update_epochs):
            el = torch.tensor(0.0)
            ec = 0
            for i in range(len(buf.mdo_obs)):
                oi = buf.mdo_obs[i]
                if oi.numel() == 0:
                    continue
                ai = torch.tensor(buf.mdo_actions[i], dtype=torch.long)
                olp = buf.mdo_log_probs[i]
                nvi = buf.mdo_num_vnfs[i]
                if nvi == 0 or i >= len(buf.mdo_tier_masks):
                    continue
                tmi = buf.mdo_tier_masks[i]
                advi = float(advs[i]) if i < len(advs) else 0.0
                nlp, ne, new_logits = policy.evaluate_actions(oi, tmi, ai, nvi)
                r = torch.exp(nlp.sum() - olp.sum())
                u = r * advi
                c = torch.clamp(r, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advi
                ppo_loss = -torch.min(u, c) - cfg.entropy_coef * ne

                # KL prior regularisation
                if kl_beta > 0 and i < len(suggested_domains_list) and suggested_domains_list[i]:
                    prior_logits = build_prior_logits(
                        suggested_domains_list[i], num_domains, tmi, temperature=prior_temp,
                    )
                    kl = analytical_kl(new_logits[:nvi], prior_logits[:nvi], tmi[:nvi])
                    kl = kl.clamp(max=10.0)
                    if not torch.isnan(kl) and not torch.isinf(kl):
                        ppo_loss = ppo_loss + kl_beta * kl

                el = el + ppo_loss
                ec += 1
            if ec > 0:
                mean_el = el / ec
                if torch.isnan(mean_el) or torch.isinf(mean_el):
                    logger.warning("    NaN/Inf in MDO loss (epoch), skipping")
                    mdo_opt.zero_grad()
                    continue
                mdo_opt.zero_grad()
                mean_el.backward()
                # Check for NaN grads before stepping
                has_nan = False
                for p in policy.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        has_nan = True
                        break
                if has_nan:
                    logger.warning("    NaN grad in MDO update, skipping step")
                    mdo_opt.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                mdo_opt.step()

        # Domain actor updates
        for domain_id, transitions in buf.domain_actor.items():
            if domain_id not in actor_opts:
                continue
            actor = actors[domain_id]
            optimizer = actor_opts[domain_id]
            for _epoch in range(cfg.update_epochs):
                epoch_loss = torch.tensor(0.0)
                epoch_count = 0
                for t in transitions:
                    if not t.steps:
                        continue
                    advantage = t.terminal_reward - critic_baseline
                    for step in t.steps:
                        new_logits = actor.policy._encode_and_score(
                            step.graph_data, step.vnf_context, step.action_mask,
                        )
                        from torch.distributions import Categorical
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
                        epoch_loss = epoch_loss + step_loss - entropy_bonus
                        epoch_count += 1
                if epoch_count > 0:
                    mean_loss = epoch_loss / epoch_count
                    optimizer.zero_grad()
                    mean_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.policy.parameters(), 0.5)
                    optimizer.step()

        rate = 100 * ep.stats.admitted / max(ep.stats.total_arrivals, 1)
        logger.info("    round %d: sample=%.1f%% (%d/%d) gated=%d",
                     rnd + 1, rate, ep.stats.admitted, ep.stats.total_arrivals,
                     ep.stats.hard_penalty_fires)

    return coord


def eval_streams(coord, seed, mode="deterministic", override_coord=None):
    """Eval on K_EVAL frozen streams, return list of (admit%, cost/slice)."""
    results = []
    for k in range(K_EVAL):
        eval_seed = seed + 9000 + k
        sub = build_substrate(seed)
        delays = build_delays(sub)
        rng = np.random.default_rng(eval_seed)
        ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
        ap.generate()
        events = list(ap.events)

        sub2 = build_substrate(seed)
        ap2 = ArrivalProcess(sub2, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
        ap2.events = list(events)
        ap2._event_idx = 0
        use_coord = override_coord if override_coord is not None else coord
        runner = EpisodeRunner(
            sub2, ap2, use_coord, delays,
            plan_builder=_greedy_plan_builder,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        ep = runner.run_episode(mdo_mode=mode)
        rate = 100 * ep.stats.admitted / max(ep.stats.total_arrivals, 1)
        cost = sum(r.total_cost for r in ep.mdo_results if r.admitted)
        mean_cost = cost / max(ep.stats.admitted, 1)
        results.append((rate, mean_cost))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="KL regularisation weight toward greedy prior")
    parser.add_argument("--prior-temp", type=float, default=1.0,
                        help="Temperature for prior logits (lower=sharper)")
    args = parser.parse_args()

    sub = build_substrate(0)
    max_vnfs = 10
    dummy_plan = PlanSummary(
        vnf_ids=["v0", "v1"], required_tiers=[InfrastructureTier.MEC] * 2,
        suggested_domains=[0, 1], cpu_demands=[1.0] * 2, ram_demands=[1.0] * 2,
        vcrs=[1.0] * 2, bw_demands=[10.0],
    )
    obs = build_mdo_observation(sub, dummy_plan)
    obs_tensor = observation_to_tensor(obs, max_vnfs=max_vnfs)
    obs_dim = obs_tensor.shape[0]
    num_domains = 3

    seeds = list(range(args.seeds))

    logger.info("=" * 70)
    logger.info("STEP A: KL PRIOR EVAL  β=%.2f  temp=%.1f", args.beta, args.prior_temp)
    logger.info("  %d seeds × %d eval streams × 4 configs", len(seeds), K_EVAL)
    logger.info("  Configs: ORION_BETA0, ORION_PRIOR(β=%.2f), RANDOM_FEAS, GREEDY",
                args.beta)
    logger.info("=" * 70)

    all_beta0 = []
    all_prior = []
    all_random = []
    all_greedy = []

    for s in seeds:
        logger.info("")
        logger.info("--- seed %d ---", s)

        # 1. ORION β=0 (baseline)
        logger.info("  training ORION β=0...")
        beta0_coord = train_policy(
            s, args.rounds, alpha=0.1, xi=0.1, eta=0.1,
            obs_dim=obs_dim, num_domains=num_domains, max_vnfs=max_vnfs,
            kl_beta=0.0,
        )
        beta0_res = eval_streams(beta0_coord, s, mode="deterministic")
        beta0_rates = [r[0] for r in beta0_res]
        beta0_costs = [r[1] for r in beta0_res]
        beta0_mean = np.mean(beta0_rates)
        all_beta0.append((beta0_mean, np.mean(beta0_costs)))
        logger.info("  BETA0:  admit=%.1f%% ± %.1f  cost=%.1f",
                     beta0_mean, np.std(beta0_rates), np.mean(beta0_costs))

        # 2. ORION β>0 with greedy prior
        logger.info("  training ORION β=%.2f (greedy prior)...", args.beta)
        prior_coord = train_policy(
            s, args.rounds, alpha=0.1, xi=0.1, eta=0.1,
            obs_dim=obs_dim, num_domains=num_domains, max_vnfs=max_vnfs,
            kl_beta=args.beta, prior_temp=args.prior_temp,
        )
        prior_res = eval_streams(prior_coord, s, mode="deterministic")
        prior_rates = [r[0] for r in prior_res]
        prior_costs = [r[1] for r in prior_res]
        prior_mean = np.mean(prior_rates)
        all_prior.append((prior_mean, np.mean(prior_costs)))
        logger.info("  PRIOR:  admit=%.1f%% ± %.1f  cost=%.1f",
                     prior_mean, np.std(prior_rates), np.mean(prior_costs))

        # 3. RANDOM FEASIBLE — no training, random MDO
        # Use beta0 actors (trained) but random MDO selection
        random_res = eval_streams(beta0_coord, s, mode="random")
        random_rates = [r[0] for r in random_res]
        random_costs = [r[1] for r in random_res]
        random_mean = np.mean(random_rates)
        all_random.append((random_mean, np.mean(random_costs)))
        logger.info("  RANDOM: admit=%.1f%% ± %.1f  cost=%.1f",
                     random_mean, np.std(random_rates), np.mean(random_costs))

        # 4. GREEDY
        greedy_coord = MDOCoordinator(
            None, beta0_coord.domain_actors,
            MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
        )
        greedy_res = eval_streams(beta0_coord, s, mode="follow_prior",
                                   override_coord=greedy_coord)
        greedy_rates = [r[0] for r in greedy_res]
        greedy_costs = [r[1] for r in greedy_res]
        greedy_mean = np.mean(greedy_rates)
        all_greedy.append((greedy_mean, np.mean(greedy_costs)))
        logger.info("  GREEDY: admit=%.1f%% ± %.1f  cost=%.1f",
                     greedy_mean, np.std(greedy_rates), np.mean(greedy_costs))

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY ACROSS %d SEEDS  (β=%.2f, temp=%.1f)",
                len(seeds), args.beta, args.prior_temp)
    logger.info("=" * 70)

    def report(label, data):
        admits = [x[0] for x in data]
        costs = [x[1] for x in data]
        logger.info("  %-20s admit = %.1f%% ± %.1f  cost = %.1f ± %.1f  [%s]",
                     label,
                     np.mean(admits), np.std(admits),
                     np.mean(costs), np.std(costs),
                     ", ".join(f"{x:.1f}" for x in admits))

    report("ORION β=0:", all_beta0)
    report(f"ORION β={args.beta}:", all_prior)
    report("RANDOM FEASIBLE:", all_random)
    report("GREEDY:", all_greedy)

    # Key comparisons
    beta0_mean_all = np.mean([x[0] for x in all_beta0])
    prior_mean_all = np.mean([x[0] for x in all_prior])
    random_mean_all = np.mean([x[0] for x in all_random])
    greedy_mean_all = np.mean([x[0] for x in all_greedy])

    beta0_std = np.std([x[0] for x in all_beta0])
    prior_std = np.std([x[0] for x in all_prior])
    random_std = np.std([x[0] for x in all_random])

    logger.info("")
    logger.info("KEY COMPARISONS:")
    logger.info("  Prior effect (β=%.2f − β=0):     %+.1f%% admission",
                args.beta, prior_mean_all - beta0_mean_all)
    logger.info("  Variance reduction:              β=0 ±%.1f → β=%.2f ±%.1f",
                beta0_std, args.beta, prior_std)
    logger.info("  Selector vs random-feasible:     %.1f%% vs %.1f%% (%+.1f%%)",
                beta0_mean_all, random_mean_all,
                beta0_mean_all - random_mean_all)
    logger.info("  Prior-to-greedy gap:             %.1f%% (%.1f vs %.1f)",
                greedy_mean_all - prior_mean_all, prior_mean_all, greedy_mean_all)

    logger.info("")
    logger.info("INTERPRETATION:")
    if prior_mean_all > beta0_mean_all + 2:
        logger.info("  Prior LIFTS admission — the plan-prior coupling works.")
        if prior_std < beta0_std - 1:
            logger.info("  Prior also TIGHTENS variance — stabilising effect confirmed.")
    elif prior_mean_all < beta0_mean_all - 2:
        logger.info("  Prior HURTS admission — coupling over-constrains the MDO.")
        logger.info("  → Try lower β or higher temperature before concluding.")
    else:
        logger.info("  Prior has NEGLIGIBLE effect on admission (within ±2%%).")
        logger.info("  → The plan is not informative enough at this β, or MDO ignores it.")

    if abs(beta0_mean_all - random_mean_all) < 3:
        logger.info("  Learned selector ≈ random-feasible — selector adds no value.")
        logger.info("  → Paper story: memory+planner carry the weight, selector is simple.")
    elif beta0_mean_all > random_mean_all + 3:
        logger.info("  Learned selector BEATS random-feasible by %.1f%%.",
                     beta0_mean_all - random_mean_all)


if __name__ == "__main__":
    main()
