#!/usr/bin/env python3
"""Multi-seed evaluation: 3 configs × 5 seeds × 5 eval streams.

Configs:
  ORION (α=0.1, ξ=0.1, η=0.1) — current reward
  PURE  (α=0,   ξ=0,   η=0)   — admission-only reward
  GREEDY (follow_prior)         — greedy FFD baseline

Also prints per-slice reward decomposition for one eval episode
to diagnose reward scale.

Usage:
    python scripts/multiseed_eval.py --rounds 10 --seeds 5
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
from orion.baselines.greedy_ffd import compute_cost_greedy, _run_greedy_ffd, GreedyConfig
from orion.actors.policy import DomainPolicy
from orion.config import TopologyConfig
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
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiseed")

NUM_ARRIVALS = 200
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
K_EVAL = 5


def build_substrate(seed):
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(num_domains=3, nodes_per_domain=[8, 10, 12],
                       intra_link_density=0.4, inter_domain_links=4), rng)


def build_delays(substrate):
    delays = {}
    for u, v in substrate.graph.edges():
        u_dom = substrate.graph.nodes[u].get("domain_id", -1)
        v_dom = substrate.graph.nodes[v].get("domain_id", -1)
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


def train_policy(seed, rounds, alpha, xi, eta, obs_dim, num_domains, max_vnfs):
    """Train one policy from scratch and return the coordinator."""
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

        # MDO policy update
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
                nlp, ne, _ = policy.evaluate_actions(oi, tmi, ai, nvi)
                r = torch.exp(nlp.sum() - olp.sum())
                u = r * advi
                c = torch.clamp(r, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advi
                el = el + (-torch.min(u, c)) - cfg.entropy_coef * ne
                ec += 1
            if ec > 0:
                mdo_opt.zero_grad()
                (el / ec).backward()
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
        logger.info("    round %d: sample=%.1f%% (%d/%d) structural=%d gated=%d departs=%d",
                     rnd + 1, rate, ep.stats.admitted, ep.stats.total_arrivals,
                     ep.stats.rejected_structural, ep.stats.hard_penalty_fires,
                     ep.stats.departures)

    return coord


def eval_streams(coord, seed, mode="deterministic", override_coord=None):
    """Eval on K_EVAL frozen streams, return list of (admit%, cost/slice).

    If override_coord is set, use that coordinator instead of coord (for greedy).
    """
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


def print_reward_decomposition(coord, seed):
    """Print per-slice reward terms for one eval episode."""
    sub = build_substrate(seed)
    delays = build_delays(sub)
    rng = np.random.default_rng(seed + 9000)
    ap = ArrivalProcess(sub, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()

    sub2 = build_substrate(seed)
    ap2 = ArrivalProcess(sub2, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap2.events = list(ap.events)
    ap2._event_idx = 0
    runner = EpisodeRunner(
        sub2, ap2, coord, delays,
        plan_builder=_greedy_plan_builder,
        reward_weights=RewardWeights(lambda_viol=10.0),
    )
    ep = runner.run_episode(mdo_mode="sample")

    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-SLICE REWARD DECOMPOSITION (one eval episode, sample mode)")
    logger.info("=" * 70)
    logger.info("  %6s  %8s  %10s  %10s  %10s  %10s  %10s  %10s",
                "status", "mu*z", "alpha*C/Cg", "eta*LS", "xi*trial", "hard_pen",
                "TOTAL", "cost/Cg")

    admitted_terms = []
    rejected_terms = []
    for r in ep.mdo_results:
        rc = r.reward
        status = "ADMIT" if r.admitted else "REJECT"
        cost_ratio = -rc.efficiency / 0.1 if rc.efficiency != 0 else 0.0

        row = (status, rc.admission, rc.efficiency, rc.quality_shaping,
               rc.trial_penalty, rc.hard_penalty, rc.total, cost_ratio)

        if r.admitted:
            admitted_terms.append(row)
        else:
            rejected_terms.append(row)

    # Print sample of admitted slices
    logger.info("  --- ADMITTED (first 15) ---")
    for row in admitted_terms[:15]:
        logger.info("  %6s  %8.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f", *row)

    logger.info("  --- REJECTED (first 10) ---")
    for row in rejected_terms[:10]:
        logger.info("  %6s  %8.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f", *row)

    if admitted_terms:
        avg_mu = np.mean([r[1] for r in admitted_terms])
        avg_eff = np.mean([r[2] for r in admitted_terms])
        avg_qs = np.mean([r[3] for r in admitted_terms])
        avg_trial = np.mean([r[4] for r in admitted_terms])
        avg_hard = np.mean([r[5] for r in admitted_terms])
        avg_total = np.mean([r[6] for r in admitted_terms])
        avg_cr = np.mean([r[7] for r in admitted_terms])
        logger.info("  --- ADMITTED AVERAGES ---")
        logger.info("  %6s  %8.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f",
                     "AVG", avg_mu, avg_eff, avg_qs, avg_trial, avg_hard, avg_total, avg_cr)
        logger.info("")
        logger.info("  mu*z / |alpha*C/Cg| = %.1f  (ratio of admission bonus to cost penalty)",
                     abs(avg_mu / avg_eff) if avg_eff != 0 else float('inf'))

    if rejected_terms:
        avg_trial_rej = np.mean([r[4] for r in rejected_terms])
        avg_total_rej = np.mean([r[6] for r in rejected_terms])
        logger.info("  rejected avg total = %.3f  (trial penalty only: %.3f)",
                     avg_total_rej, avg_trial_rej)

    logger.info("")
    logger.info("  net(admit) - net(reject) = %.3f  (incentive to admit)",
                avg_total - avg_total_rej if admitted_terms and rejected_terms else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    # Probe dimensions
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

    # ── Reward decomposition (first seed only, before multi-seed) ──────
    logger.info("Training ORION (α=0.1) seed=0 for reward decomposition...")
    coord_decomp = train_policy(0, args.rounds, alpha=0.1, xi=0.1, eta=0.1,
                                 obs_dim=obs_dim, num_domains=num_domains, max_vnfs=max_vnfs)
    print_reward_decomposition(coord_decomp, seed=0)

    # ── Multi-seed evaluation ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("MULTI-SEED EVALUATION: %d seeds × %d streams × 3 configs",
                len(seeds), K_EVAL)
    logger.info("=" * 70)

    all_orion = []
    all_pure = []
    all_greedy = []

    for s in seeds:
        logger.info("")
        logger.info("--- seed %d ---", s)

        # ORION (α=0.1)
        logger.info("  training ORION (α=0.1)...")
        orion_coord = train_policy(s, args.rounds, alpha=0.1, xi=0.1, eta=0.1,
                                    obs_dim=obs_dim, num_domains=num_domains, max_vnfs=max_vnfs)
        orion_res = eval_streams(orion_coord, s, mode="deterministic")
        orion_rates = [r[0] for r in orion_res]
        orion_costs = [r[1] for r in orion_res]
        orion_mean = np.mean(orion_rates)
        all_orion.append((orion_mean, np.mean(orion_costs)))
        logger.info("  ORION: admit=%.1f%% ± %.1f  cost=%.1f",
                     orion_mean, np.std(orion_rates), np.mean(orion_costs))

        # PURE (α=0)
        logger.info("  training PURE (α=0)...")
        pure_coord = train_policy(s, args.rounds, alpha=0.0, xi=0.0, eta=0.0,
                                   obs_dim=obs_dim, num_domains=num_domains, max_vnfs=max_vnfs)
        pure_res = eval_streams(pure_coord, s, mode="deterministic")
        pure_rates = [r[0] for r in pure_res]
        pure_costs = [r[1] for r in pure_res]
        pure_mean = np.mean(pure_rates)
        all_pure.append((pure_mean, np.mean(pure_costs)))
        logger.info("  PURE:  admit=%.1f%% ± %.1f  cost=%.1f",
                     pure_mean, np.std(pure_rates), np.mean(pure_costs))

        # GREEDY — use trained actors from ORION but no MDO policy, n_part=1
        greedy_coord = MDOCoordinator(
            None, orion_coord.domain_actors,
            MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
        )
        greedy_res = eval_streams(orion_coord, s, mode="follow_prior", override_coord=greedy_coord)
        greedy_rates = [r[0] for r in greedy_res]
        greedy_costs = [r[1] for r in greedy_res]
        greedy_mean = np.mean(greedy_rates)
        all_greedy.append((greedy_mean, np.mean(greedy_costs)))
        logger.info("  GREEDY: admit=%.1f%% ± %.1f  cost=%.1f",
                     greedy_mean, np.std(greedy_rates), np.mean(greedy_costs))

    # ── Summary with error bars ─────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY ACROSS %d SEEDS", len(seeds))
    logger.info("=" * 70)

    orion_admits = [x[0] for x in all_orion]
    pure_admits = [x[0] for x in all_pure]
    greedy_admits = [x[0] for x in all_greedy]
    orion_cost_means = [x[1] for x in all_orion]
    pure_cost_means = [x[1] for x in all_pure]
    greedy_cost_means = [x[1] for x in all_greedy]

    logger.info("  ORION (α=0.1):  admit = %.1f%% ± %.1f  cost = %.1f ± %.1f  [per seed: %s]",
                np.mean(orion_admits), np.std(orion_admits),
                np.mean(orion_cost_means), np.std(orion_cost_means),
                ", ".join(f"{x:.1f}" for x in orion_admits))
    logger.info("  PURE  (α=0):    admit = %.1f%% ± %.1f  cost = %.1f ± %.1f  [per seed: %s]",
                np.mean(pure_admits), np.std(pure_admits),
                np.mean(pure_cost_means), np.std(pure_cost_means),
                ", ".join(f"{x:.1f}" for x in pure_admits))
    logger.info("  GREEDY:         admit = %.1f%% ± %.1f  cost = %.1f ± %.1f  [per seed: %s]",
                np.mean(greedy_admits), np.std(greedy_admits),
                np.mean(greedy_cost_means), np.std(greedy_cost_means),
                ", ".join(f"{x:.1f}" for x in greedy_admits))

    logger.info("")
    gap_orion_greedy = np.mean(greedy_admits) - np.mean(orion_admits)
    gap_pure_greedy = np.mean(greedy_admits) - np.mean(pure_admits)
    logger.info("  ORION-to-greedy gap: %.1f%%", gap_orion_greedy)
    logger.info("  PURE-to-greedy gap:  %.1f%%", gap_pure_greedy)
    logger.info("  Cost-terms effect:   %.1f%% (PURE - ORION)", np.mean(pure_admits) - np.mean(orion_admits))

    logger.info("")
    logger.info("DECISION RULE:")
    logger.info("  If alpha*Cost >> mu*z in decomposition → rescale or Lagrangian")
    logger.info("  If PURE ≈ greedy across seeds → reward is the fix, shelve decoder")
    logger.info("  If PURE < greedy-5%% across seeds → representation matters, revisit decoder")


if __name__ == "__main__":
    main()
