#!/usr/bin/env python3
"""First real training run: CTDE-PPO with Agent B prior.

Assembles the full stack:
  - SubstrateNetwork (3-domain, locked topology)
  - ArrivalProcess (2000 arrivals per episode)
  - Agent B via llama-cpp OpenAI-compatible server (no K^B, Tele-it fine-tuning carries domain knowledge)
  - MDO policy + domain actors + centralised critic
  - CTDE-PPO joint training loop
  - Quality tracker + greedy baseline comparison

Usage:
    # Ensure llama-cpp server is running on port 8000:
    # .venv/bin/python -m llama_cpp.server --model models/LLama-3-8B-Tele-it.Q4_K_M.gguf \
    #   --n_threads 32 --n_ctx 4096 --host 0.0.0.0 --port 8000 --chat_format llama-3

    python scripts/run_agent_b_training.py --seed 0 --rounds 20
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import greedy_place_on_substrate
from orion.config import TopologyConfig
from orion.llm.agent_b_planner import (
    PlanQualityTracker,
    make_constrained_plan_builder,
)
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.observation import build_mdo_observation, observation_to_tensor
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.sim.arrival_process import ArrivalProcess, EventType
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.critic import CentralisedCritic
from orion.training.gae import compute_gae
from orion.training.global_state import (
    GlobalStateStats,
    encode_global_state,
    probe_global_state_dim,
)
from orion.training.kl_schedule import beta_linear
from orion.training.ppo_update import ppo_mdo_update
from orion.types import InfrastructureTier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("training")

# ── Locked defaults ──────────────────────────────────────────────────────────

NUM_ARRIVALS = 2000
ARRIVAL_RATE = 4.0
SERVICE_RATE = 0.02
MAX_VNFS = 8
HIDDEN_DIM = 64
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
MAX_GRAD_NORM = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95
PRIOR_TEMPERATURE = 1.0


# ── Setup helpers ────────────────────────────────────────────────────────────


def build_substrate(seed: int) -> SubstrateNetwork:
    rng = np.random.default_rng(seed)
    return generate_multi_domain_topology(
        TopologyConfig(num_domains=3, nodes_per_domain=[8, 10, 12],
                       intra_link_density=0.4, inter_domain_links=4), rng)


def build_arrival_process(substrate, seed):
    rng = np.random.default_rng(seed + 1_000_000)
    ap = ArrivalProcess(substrate, NUM_ARRIVALS, ARRIVAL_RATE, SERVICE_RATE, rng)
    ap.generate()
    return ap


def build_delays(substrate):
    g = substrate.graph
    delays = {}
    for u, v, d in g.edges(data=True):
        sd, dd = g.nodes[u]["domain_id"], g.nodes[v]["domain_id"]
        if sd != dd:
            delays[(sd, dd)] = min(delays.get((sd, dd), float("inf")), d["propagation_delay"])
    return delays


def probe_obs_dim(substrate):
    plan = PlanSummary(
        vnf_ids=[f"d{i}" for i in range(MAX_VNFS)],
        required_tiers=[InfrastructureTier.MEC] * MAX_VNFS,
        suggested_domains=[0] * MAX_VNFS,
        cpu_demands=[1.0] * MAX_VNFS,
        ram_demands=[1.0] * MAX_VNFS,
        vcrs=[1.0] * MAX_VNFS,
        bw_demands=[10.0] * (MAX_VNFS - 1),
    )
    obs = observation_to_tensor(build_mdo_observation(substrate, plan))
    return obs.shape[0]


def pad_obs(obs, target_dim):
    if obs.shape[-1] >= target_dim:
        return obs[..., :target_dim]
    pad_size = target_dim - obs.shape[-1]
    return torch.cat([obs, torch.zeros(*obs.shape[:-1], pad_size)], dim=-1)


class PaddedMDOPolicy(MDOPolicy):
    def __init__(self, obs_dim, **kwargs):
        super().__init__(obs_dim=obs_dim, **kwargs)
        self._fixed_obs_dim = obs_dim

    def forward(self, obs, tier_mask, num_vnfs, deterministic=False):
        return super().forward(pad_obs(obs, self._fixed_obs_dim), tier_mask, num_vnfs, deterministic)

    def evaluate_actions(self, obs, tier_mask, actions, num_vnfs):
        return super().evaluate_actions(pad_obs(obs, self._fixed_obs_dim), tier_mask, actions, num_vnfs)

    def get_value(self, obs):
        return super().get_value(pad_obs(obs, self._fixed_obs_dim))


def run_greedy_episode(substrate, arrival_process):
    substrate.reset()
    arrival_process.reset()
    total = admitted = 0
    while arrival_process.has_next():
        event = arrival_process.next_event()
        if event.event_type == EventType.DEPARTURE:
            plan = substrate._active_slices.get(event.request_id)
            if plan:
                substrate.deallocate(plan[0], plan[1])
            continue
        total += 1
        result = greedy_place_on_substrate(substrate, event.slice_request)
        if result.feasible:
            admitted += 1
    return {"total": total, "admitted": admitted,
            "rate": admitted / total if total > 0 else 0.0}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--llm-model", default="models/LLama-3-8B-Tele-it.Q4_K_M.gguf")
    parser.add_argument("--run-dir", default="runs/agent_b_seed0")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed + 2_000_000)

    # ── Build stack ──────────────────────────────────────────────────
    substrate = build_substrate(seed)
    arrival_process = build_arrival_process(substrate, seed)
    delays = build_delays(substrate)
    obs_dim = probe_obs_dim(substrate)
    num_domains = substrate.num_domains

    # Agent B (constrained JSON-mode decoding — 0% fallback on all chain lengths)
    quality_tracker = PlanQualityTracker()
    plan_builder = make_constrained_plan_builder(
        args.llm_model, n_threads=32,
        quality_tracker=quality_tracker,
    )

    # MDO + actors
    mdo_policy = PaddedMDOPolicy(
        obs_dim=obs_dim, num_domains=num_domains,
        max_vnfs=MAX_VNFS, hidden_dim=HIDDEN_DIM, num_layers=2,
    )
    domain_actors = {}
    for d in range(num_domains):
        torch.manual_seed(seed + 3_000_000 + d)
        domain_actors[d] = DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=HIDDEN_DIM))

    coordinator = MDOCoordinator(
        policy=mdo_policy, domain_actors=domain_actors,
        config=MDOConfig(n_part=3, max_inter_domain_hops=3),
    )

    runner = EpisodeRunner(
        substrate=substrate, arrival_process=arrival_process,
        coordinator=coordinator, inter_domain_delays=delays,
        plan_builder=plan_builder,
    )

    # Critic + optimizers
    critic_dim = probe_global_state_dim(substrate)
    critic = CentralisedCritic(critic_dim, HIDDEN_DIM, 2)
    actor_opt = torch.optim.Adam(mdo_policy.parameters(), lr=LR_ACTOR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    domain_opts = {
        d: torch.optim.Adam(domain_actors[d].policy.parameters(), lr=LR_ACTOR)
        for d in range(num_domains)
    }

    # ── Greedy baseline ──────────────────────────────────────────────
    logger.info("Running greedy baseline...")
    greedy_sub = build_substrate(seed)
    greedy_ap = build_arrival_process(greedy_sub, seed)
    greedy_stats = run_greedy_episode(greedy_sub, greedy_ap)
    logger.info("Greedy: %d/%d (%.1f%%)", greedy_stats["admitted"],
                greedy_stats["total"], greedy_stats["rate"] * 100)

    # ── Training loop ────────────────────────────────────────────────
    from torch.distributions import Categorical
    from orion.actors.policy import DomainPolicy as DP

    global_step = 0
    total_steps = args.rounds * NUM_ARRIVALS
    round_logs = []

    logger.info("Starting training: %d rounds, %d arrivals/round, seed=%d",
                args.rounds, NUM_ARRIVALS, seed)

    for round_idx in range(args.rounds):
        t0 = time.time()

        # Collect episode
        runner.reset()
        episode = runner.run_episode(mdo_mode="sample")

        # Build PPO buffer
        buffer = PPORolloutBuffer()
        stats = GlobalStateStats(
            episode.stats.total_arrivals, episode.stats.admitted,
            episode.stats.rejected_by_mdo,
        )
        gs = encode_global_state(substrate, stats)
        with torch.no_grad():
            cv = float(critic(gs).item())

        for t in episode.rollout.mdo:
            buffer.append_mdo(
                mdo_obs=t.obs, action=t.action, log_prob=t.log_probs,
                entropy=t.entropy, aux_value=t.value_estimate,
                global_state=gs, critic_value=cv,
                reward=t.terminal_reward, done=t.committed,
            )
        for did, ts in episode.rollout.domain_actor.items():
            for t in ts:
                buffer.append_domain_actor(t)

        if len(buffer) == 0:
            global_step += episode.stats.total_arrivals
            continue

        # GAE
        rewards = buffer.reward_tensor()
        values = buffer.value_tensor(bootstrap=0.0)
        dones = buffer.done_tensor()
        advantages, returns = compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.set_gae(advantages, returns)

        # ── Critic update ────────────────────────────────────────────
        critic_loss_sum = 0.0
        for _epoch in range(args.epochs):
            gs_stack = torch.stack(buffer.global_states)
            ret = buffer.returns.detach()
            old_vals = torch.tensor(buffer.critic_values, dtype=torch.float32)
            new_vals = critic(gs_stack).squeeze(-1)
            v_clipped = old_vals + torch.clamp(new_vals - old_vals, -CLIP_EPS, CLIP_EPS)
            raw_loss = (new_vals - ret) ** 2
            clip_loss = (v_clipped - ret) ** 2
            vloss = 0.5 * torch.max(raw_loss, clip_loss).mean()
            critic_opt.zero_grad()
            (VALUE_LOSS_COEF * vloss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            critic_opt.step()
            critic_loss_sum += vloss.item()

        # ── Domain actor CTDE-PPO update ─────────────────────────────
        critic_baseline = cv  # detached float
        actor_loss_sum = 0.0
        actor_step_count = 0

        for did, transitions in buffer.domain_actor.items():
            if did not in domain_opts or not transitions:
                continue
            actor = domain_actors[did]
            opt = domain_opts[did]

            for _epoch in range(args.epochs):
                epoch_loss = torch.zeros(1)
                n = 0
                for t in transitions:
                    advantage = t.terminal_reward - critic_baseline
                    for step in t.steps:
                        new_logits = actor.policy._encode_and_score(
                            step.graph_data, step.vnf_context, step.action_mask,
                        )
                        dist = Categorical(logits=new_logits)
                        n_nodes = step.action_mask.size(0)
                        a = torch.tensor(n_nodes if step.action_idx == DP.NULL_ACTION else step.action_idx)
                        new_lp = dist.log_prob(a)
                        ratio = torch.exp(new_lp - step.log_prob)
                        unclipped = ratio * advantage
                        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantage
                        epoch_loss = epoch_loss - torch.min(unclipped, clipped)
                        n += 1
                if n == 0:
                    continue
                loss = epoch_loss / n - ENTROPY_COEF * sum(
                    step.entropy for t in transitions for step in t.steps
                ) / max(n, 1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.policy.parameters(), MAX_GRAD_NORM)
                opt.step()
                actor_loss_sum += loss.item()
                actor_step_count += n

        global_step += len(buffer)
        elapsed = time.time() - t0

        # ── Log ──────────────────────────────────────────────────────
        rl = {
            "round": round_idx,
            "arrivals": episode.stats.total_arrivals,
            "admitted": episode.stats.admitted,
            "rejected_mdo": episode.stats.rejected_by_mdo,
            "rejected_structural": episode.stats.rejected_structural,
            "acceptance_rate": episode.stats.admission_rate,
            "cumulative_reward": episode.stats.cumulative_reward,
            "critic_loss": critic_loss_sum / args.epochs,
            "actor_loss": actor_loss_sum / max(actor_step_count, 1),
            "buffer_size": len(buffer),
            "wall_time": elapsed,
            "per_type_admitted": dict(episode.stats.per_slice_type_admitted),
            "per_type_total": dict(episode.stats.per_slice_type_total),
        }
        round_logs.append(rl)

        logger.info(
            "round %2d/%d | admit %d/%d (%.1f%%) | rew %.1f | "
            "critic_loss %.3f | actor_loss %.4f | %.0fs",
            round_idx + 1, args.rounds,
            episode.stats.admitted, episode.stats.total_arrivals,
            episode.stats.admission_rate * 100,
            episode.stats.cumulative_reward,
            rl["critic_loss"], rl["actor_loss"], elapsed,
        )

    # ── Save results ─────────────────────────────────────────────────
    quality_summary = quality_tracker.summary()
    logger.info("Agent B quality: %s", json.dumps(quality_summary, indent=2))

    output = {
        "seed": seed,
        "rounds": args.rounds,
        "greedy": greedy_stats,
        "quality": quality_summary,
        "round_logs": round_logs,
    }
    out_path = run_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
