#!/usr/bin/env python3
"""Multi-objective profile: admission, cost, cross-domain splits, delay.

Tests whether random-feasible's high admission is Pareto-dominated
(high cost + many cross-domain splits) or genuinely dominant.

Configs:
  ORION_BETA0    — learned MDO selector, β=0
  RANDOM_FEAS    — random-feasible MDO, no learning
  GREEDY         — greedy FFD baseline
  COST_AWARE_LB  — least-loaded feasible domain, cost+locality tiebreak
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
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


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def _cost_aware_lb_plan_builder(slice_req, substrate):
    """Least-loaded feasible domain with locality tiebreak.

    For each VNF: find all feasible domains (tier match + enough CPU/RAM on
    at least one node), then pick the one with the most total residual CPU,
    breaking ties by preferring the domain where the most other VNFs in this
    slice are already assigned (locality) then lowest domain_id.
    """
    g = substrate.graph

    # Gather per-domain residual CPU and feasible tiers
    domain_cpu: dict[int, float] = {}
    domain_nodes_by_tier: dict[int, dict[str, list[str]]] = {}
    for node_id, d in g.nodes(data=True):
        dom = d.get("domain_id", -1)
        if dom < 0:
            continue
        domain_cpu[dom] = domain_cpu.get(dom, 0.0) + float(d["cpu_residual"])
        tier = d["tier"]
        domain_nodes_by_tier.setdefault(dom, {}).setdefault(tier, []).append(node_id)

    vnf_ids, required_tiers, suggested_domains = [], [], []

    # Track how many VNFs assigned to each domain for locality
    domain_counts: dict[int, int] = {}

    for vnf in slice_req.vnfs:
        vnf_ids.append(vnf.vnf_id)

        # Determine required tier from permitted_nodes
        req_tier = None
        if vnf.permitted_nodes:
            for nid in vnf.permitted_nodes:
                if nid in g.nodes:
                    req_tier = g.nodes[nid]["tier"]
                    break
        if req_tier is None:
            req_tier = "MEC"
        required_tiers.append(InfrastructureTier(req_tier))

        # Find feasible domains: have at least one node with matching tier
        # and enough CPU+RAM
        feasible_domains = []
        for dom, nodes_by_tier in domain_nodes_by_tier.items():
            tier_nodes = nodes_by_tier.get(req_tier, [])
            for nid in tier_nodes:
                nd = g.nodes[nid]
                if float(nd["cpu_residual"]) >= vnf.cpu_demand and \
                   float(nd["ram_residual"]) >= vnf.ram_demand:
                    feasible_domains.append(dom)
                    break

        if not feasible_domains:
            # Fallback: any domain
            feasible_domains = list(domain_cpu.keys())

        # Pick domain: most residual CPU (load-balance), break ties by
        # locality (most VNFs already there), then lowest domain_id
        best = min(
            feasible_domains,
            key=lambda d: (
                -domain_cpu.get(d, 0.0),           # most residual CPU first
                -domain_counts.get(d, 0),           # locality tiebreak
                d,                                   # stable ordering
            ),
        )
        suggested_domains.append(best)
        domain_counts[best] = domain_counts.get(best, 0) + 1

    return PlanSummary(
        vnf_ids=vnf_ids, required_tiers=required_tiers,
        suggested_domains=suggested_domains,
        cpu_demands=[v.cpu_demand for v in slice_req.vnfs],
        ram_demands=[v.ram_demand for v in slice_req.vnfs],
        vcrs=[v.vcr for v in slice_req.vnfs],
        bw_demands=[f.bandwidth_demand for f in slice_req.flow_edges],
    )


# ── Profile metrics ────────────────────────────────────────────────────────

@dataclass
class ProfileMetrics:
    admission_rate: float = 0.0
    cost_per_slice: float = 0.0
    cross_domain_splits: float = 0.0  # avg unique domains per admitted slice
    avg_e2e_delay: float = 0.0
    hard_violations: int = 0
    total_admitted: int = 0
    total_arrivals: int = 0
    total_cost: float = 0.0


def extract_profile(ep) -> ProfileMetrics:
    """Extract multi-objective profile from an EpisodeResult."""
    m = ProfileMetrics()
    m.total_arrivals = ep.stats.total_arrivals
    m.total_admitted = ep.stats.admitted
    m.admission_rate = 100 * ep.stats.admitted / max(ep.stats.total_arrivals, 1)
    m.hard_violations = ep.stats.hard_penalty_fires

    costs = []
    delays = []
    domain_counts = []

    for r in ep.mdo_results:
        if not r.admitted:
            continue
        costs.append(r.total_cost)
        delays.append(r.e2e_delay)
        if r.partition is not None:
            domain_counts.append(len(set(r.partition)))

    m.total_cost = sum(costs)
    m.cost_per_slice = np.mean(costs) if costs else 0.0
    m.avg_e2e_delay = np.mean(delays) if delays else 0.0
    m.cross_domain_splits = np.mean(domain_counts) if domain_counts else 0.0

    return m


# ── Training ────────────────────────────────────────────────────────────────

def train_policy(seed, rounds, alpha, xi, eta, obs_dim, num_domains, max_vnfs):
    """Train one policy from scratch (β=0)."""
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
                el = el + ppo_loss
                ec += 1
            if ec > 0:
                mean_el = el / ec
                if torch.isnan(mean_el) or torch.isinf(mean_el):
                    mdo_opt.zero_grad()
                    continue
                mdo_opt.zero_grad()
                mean_el.backward()
                has_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in policy.parameters()
                )
                if has_nan:
                    mdo_opt.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                mdo_opt.step()

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


# ── Evaluation ──────────────────────────────────────────────────────────────

def eval_profile(coord, seed, mode="deterministic", plan_builder=None,
                 override_coord=None) -> ProfileMetrics:
    """Run K_EVAL streams and return averaged ProfileMetrics."""
    all_metrics = []
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
        pb = plan_builder if plan_builder is not None else _greedy_plan_builder
        runner = EpisodeRunner(
            sub2, ap2, use_coord, delays,
            plan_builder=pb,
            reward_weights=RewardWeights(lambda_viol=10.0),
        )
        ep = runner.run_episode(mdo_mode=mode)
        all_metrics.append(extract_profile(ep))

    avg = ProfileMetrics()
    avg.admission_rate = np.mean([m.admission_rate for m in all_metrics])
    avg.cost_per_slice = np.mean([m.cost_per_slice for m in all_metrics])
    avg.cross_domain_splits = np.mean([m.cross_domain_splits for m in all_metrics])
    avg.avg_e2e_delay = np.mean([m.avg_e2e_delay for m in all_metrics])
    avg.hard_violations = int(np.mean([m.hard_violations for m in all_metrics]))
    avg.total_admitted = int(np.mean([m.total_admitted for m in all_metrics]))
    avg.total_cost = np.mean([m.total_cost for m in all_metrics])
    return avg


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
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
    logger.info("MULTI-OBJECTIVE PROFILE")
    logger.info("  %d seeds × %d eval streams × 5 configs", len(seeds), K_EVAL)
    logger.info("  Configs: LEARNED, RANDOM_FEAS, GREEDY, COST_AWARE_LB")
    logger.info("=" * 70)

    # Per-seed results: list of dicts with all metrics
    config_names = ["LEARNED", "RANDOM_FEAS", "GREEDY", "COST_AWARE_LB"]
    all_results = {name: [] for name in config_names}

    for s in seeds:
        logger.info("")
        logger.info("--- seed %d ---", s)

        # 1. LEARNED (β=0)
        logger.info("  training learned selector...")
        coord = train_policy(
            s, args.rounds, alpha=0.1, xi=0.1, eta=0.1,
            obs_dim=obs_dim, num_domains=num_domains, max_vnfs=max_vnfs,
        )
        m = eval_profile(coord, s, mode="deterministic")
        all_results["LEARNED"].append(m)
        logger.info("  LEARNED:       admit=%.1f%%  cost/slice=%.1f  domains/slice=%.2f  delay=%.2f  viols=%d",
                     m.admission_rate, m.cost_per_slice, m.cross_domain_splits,
                     m.avg_e2e_delay, m.hard_violations)

        # 2. RANDOM FEASIBLE
        m = eval_profile(coord, s, mode="random")
        all_results["RANDOM_FEAS"].append(m)
        logger.info("  RANDOM_FEAS:   admit=%.1f%%  cost/slice=%.1f  domains/slice=%.2f  delay=%.2f  viols=%d",
                     m.admission_rate, m.cost_per_slice, m.cross_domain_splits,
                     m.avg_e2e_delay, m.hard_violations)

        # 3. GREEDY FFD
        greedy_coord = MDOCoordinator(
            None, coord.domain_actors,
            MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
        )
        m = eval_profile(coord, s, mode="follow_prior",
                         override_coord=greedy_coord)
        all_results["GREEDY"].append(m)
        logger.info("  GREEDY:        admit=%.1f%%  cost/slice=%.1f  domains/slice=%.2f  delay=%.2f  viols=%d",
                     m.admission_rate, m.cost_per_slice, m.cross_domain_splits,
                     m.avg_e2e_delay, m.hard_violations)

        # 4. COST-AWARE LOAD-BALANCING
        calb_coord = MDOCoordinator(
            None, coord.domain_actors,
            MDOConfig(n_part=1, mu=1.0, alpha=0.1, xi=0.1, eta=0.1),
        )
        m = eval_profile(coord, s, mode="follow_prior",
                         plan_builder=_cost_aware_lb_plan_builder,
                         override_coord=calb_coord)
        all_results["COST_AWARE_LB"].append(m)
        logger.info("  COST_AWARE_LB: admit=%.1f%%  cost/slice=%.1f  domains/slice=%.2f  delay=%.2f  viols=%d",
                     m.admission_rate, m.cost_per_slice, m.cross_domain_splits,
                     m.avg_e2e_delay, m.hard_violations)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY ACROSS %d SEEDS", len(seeds))
    logger.info("=" * 70)
    logger.info("")
    logger.info("%-16s  %8s  %10s  %12s  %8s  %6s",
                "CONFIG", "ADMIT%", "COST/SLICE", "DOMAINS/SLICE", "DELAY", "VIOLS")
    logger.info("-" * 70)

    summary = {}
    for name in config_names:
        metrics = all_results[name]
        admits = [m.admission_rate for m in metrics]
        costs = [m.cost_per_slice for m in metrics]
        domains = [m.cross_domain_splits for m in metrics]
        delays = [m.avg_e2e_delay for m in metrics]
        viols = [m.hard_violations for m in metrics]

        summary[name] = {
            "admit_mean": np.mean(admits), "admit_std": np.std(admits),
            "cost_mean": np.mean(costs), "cost_std": np.std(costs),
            "domains_mean": np.mean(domains), "domains_std": np.std(domains),
            "delay_mean": np.mean(delays), "delay_std": np.std(delays),
            "viols_mean": np.mean(viols),
        }
        s = summary[name]
        logger.info("%-16s  %5.1f±%.1f  %7.1f±%.1f  %9.2f±%.2f  %5.2f±%.2f  %5.1f",
                     name,
                     s["admit_mean"], s["admit_std"],
                     s["cost_mean"], s["cost_std"],
                     s["domains_mean"], s["domains_std"],
                     s["delay_mean"], s["delay_std"],
                     s["viols_mean"])
        logger.info("%16s  per-seed admits: [%s]", "",
                     ", ".join(f"{a:.1f}" for a in admits))

    # ── Pareto analysis ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("PARETO ANALYSIS:")
    rf = summary["RANDOM_FEAS"]
    gr = summary["GREEDY"]
    lb = summary["COST_AWARE_LB"]
    le = summary["LEARNED"]

    logger.info("  Random-feasible vs greedy:")
    logger.info("    Admission: %+.1f%%  Cost: %+.1f  Domains: %+.2f  Delay: %+.2f",
                rf["admit_mean"] - gr["admit_mean"],
                rf["cost_mean"] - gr["cost_mean"],
                rf["domains_mean"] - gr["domains_mean"],
                rf["delay_mean"] - gr["delay_mean"])

    logger.info("  Random-feasible vs cost-aware LB:")
    logger.info("    Admission: %+.1f%%  Cost: %+.1f  Domains: %+.2f  Delay: %+.2f",
                rf["admit_mean"] - lb["admit_mean"],
                rf["cost_mean"] - lb["cost_mean"],
                rf["domains_mean"] - lb["domains_mean"],
                rf["delay_mean"] - lb["delay_mean"])

    logger.info("  Cost-aware LB vs greedy:")
    logger.info("    Admission: %+.1f%%  Cost: %+.1f  Domains: %+.2f  Delay: %+.2f",
                lb["admit_mean"] - gr["admit_mean"],
                lb["cost_mean"] - gr["cost_mean"],
                lb["domains_mean"] - gr["domains_mean"],
                lb["delay_mean"] - gr["delay_mean"])

    logger.info("  Learned vs cost-aware LB:")
    logger.info("    Admission: %+.1f%%  Cost: %+.1f  Domains: %+.2f  Delay: %+.2f",
                le["admit_mean"] - lb["admit_mean"],
                le["cost_mean"] - lb["cost_mean"],
                le["domains_mean"] - lb["domains_mean"],
                le["delay_mean"] - lb["delay_mean"])

    # Is random-feasible Pareto-dominated?
    rf_dominated = (
        rf["cost_mean"] > gr["cost_mean"] and
        rf["domains_mean"] > gr["domains_mean"]
    )
    logger.info("")
    if rf_dominated:
        logger.info("  VERDICT: Random-feasible IS Pareto-dominated (high admit, high cost+splits).")
        logger.info("  → Multi-objective value proposition is viable.")
        logger.info("  → The job: match random-feasible admission at greedy-or-better cost and locality.")
    else:
        logger.info("  VERDICT: Random-feasible is NOT Pareto-dominated.")
        logger.info("  → Random-feasible achieves high admission without paying in cost or locality.")
        logger.info("  → Rethink the value proposition.")


if __name__ == "__main__":
    main()
