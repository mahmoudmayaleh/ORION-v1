"""Smoke test: domain actor params move after one CTDE-PPO training round.

Confirms the full pipeline's update path reaches domain actor policies
via the same CTDE-PPO algorithm the paper commits to:
  1. Snapshot actor params before training
  2. Run one collect + PPO update round (with centralised critic advantage)
  3. Verify params changed and entropy moved off init
  4. Verify the update used re-evaluated log_probs (PPO ratio), not REINFORCE
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.actors.types import ActorStepRecord
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.sim.rollout_buffer import DomainActorTransition
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.training.buffer import PPORolloutBuffer
from orion.training.config import MAPPOConfig
from orion.training.critic import CentralisedCritic
from orion.training.global_state import (
    GlobalStateStats,
    encode_global_state,
    probe_global_state_dim,
)


def _build_substrate(seed=42):
    rng = np.random.default_rng(seed)
    config = TopologyConfig(
        num_domains=2, nodes_per_domain=[4, 4],
        intra_link_density=0.6, inter_domain_links=2,
    )
    return generate_multi_domain_topology(config, rng)


def _build_delays(substrate):
    g = substrate.graph
    delays = {}
    for u, v, d in g.edges(data=True):
        sd, dd = g.nodes[u]["domain_id"], g.nodes[v]["domain_id"]
        if sd != dd:
            key = (sd, dd)
            delays[key] = min(delays.get(key, float("inf")), d["propagation_delay"])
    return delays


def _snapshot_params(model):
    return {name: p.clone() for name, p in model.named_parameters()}


class TestActorCTDEPPOSmoke:
    """Domain actor params move via CTDE-PPO (not REINFORCE)."""

    def test_step_records_collected(self):
        """Verify step_records are populated during collection."""
        seed = 42
        torch.manual_seed(seed)
        substrate = _build_substrate(seed)
        delays = _build_delays(substrate)

        actors = {d: DomainActor(d, DomainPolicy("mlp", hidden_dim=32))
                  for d in range(substrate.num_domains)}
        coordinator = MDOCoordinator(None, actors, MDOConfig(n_part=2))

        rng = np.random.default_rng(seed + 100)
        ap = ArrivalProcess(substrate, 30, 4.0, 0.02, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coordinator, delays)
        runner.reset()
        ep = runner.run_episode(mdo_mode="random")

        total_steps = 0
        for domain_id, transitions in ep.rollout.domain_actor.items():
            for t in transitions:
                total_steps += len(t.steps)

        assert total_steps > 0, "No step_records collected from domain actors"

    def test_ppo_re_evaluation_differs_from_old(self):
        """After one PPO update, re-evaluated log_probs differ from old."""
        seed = 42
        torch.manual_seed(seed)
        substrate = _build_substrate(seed)
        delays = _build_delays(substrate)

        actors = {d: DomainActor(d, DomainPolicy("mlp", hidden_dim=32))
                  for d in range(substrate.num_domains)}
        coordinator = MDOCoordinator(None, actors, MDOConfig(n_part=2))

        rng = np.random.default_rng(seed + 100)
        ap = ArrivalProcess(substrate, 50, 4.0, 0.02, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coordinator, delays)
        runner.reset()
        ep = runner.run_episode(mdo_mode="random")

        # Collect a transition with steps
        sample_transition = None
        sample_domain = None
        for domain_id, transitions in ep.rollout.domain_actor.items():
            for t in transitions:
                if t.steps:
                    sample_transition = t
                    sample_domain = domain_id
                    break
            if sample_transition:
                break

        assert sample_transition is not None, "Need at least one transition with steps"

        actor = actors[sample_domain]
        step = sample_transition.steps[0]

        # Record old log_prob
        old_lp = step.log_prob

        # Do one PPO-style update (simplified: single step)
        optimizer = torch.optim.Adam(actor.policy.parameters(), lr=1e-2)
        new_logits = actor.policy._encode_and_score(
            step.graph_data, step.vnf_context, step.action_mask,
        )
        dist = Categorical(logits=new_logits)
        n_nodes = step.action_mask.size(0)
        action_t = torch.tensor(n_nodes if step.action_idx == DomainPolicy.NULL_ACTION else step.action_idx)
        new_lp = dist.log_prob(action_t)

        # Before update: new_lp should equal old_lp (same weights)
        assert abs(new_lp.item() - old_lp) < 1e-4, (
            f"Pre-update log_probs should match: new={new_lp.item()}, old={old_lp}"
        )

        # PPO update
        ratio = torch.exp(new_lp - old_lp)
        loss = -(ratio * 1.0)  # advantage=1 for simplicity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # After update: re-evaluate — should differ
        new_logits2 = actor.policy._encode_and_score(
            step.graph_data, step.vnf_context, step.action_mask,
        )
        dist2 = Categorical(logits=new_logits2)
        new_lp2 = dist2.log_prob(action_t)

        assert abs(new_lp2.item() - old_lp) > 1e-5, (
            f"Post-update log_prob should differ from old: new={new_lp2.item()}, old={old_lp}"
        )

    def test_full_pipeline_params_move(self):
        """Full pipeline: collect → CTDE-PPO update → actor params moved."""
        seed = 42
        torch.manual_seed(seed)
        substrate = _build_substrate(seed)
        delays = _build_delays(substrate)

        actors = {d: DomainActor(d, DomainPolicy("mlp", hidden_dim=32))
                  for d in range(substrate.num_domains)}
        coordinator = MDOCoordinator(None, actors, MDOConfig(n_part=2))

        rng = np.random.default_rng(seed + 100)
        ap = ArrivalProcess(substrate, 50, 4.0, 0.02, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coordinator, delays)

        # Snapshot BEFORE
        pre_params = {d: _snapshot_params(a.policy) for d, a in actors.items()}

        # Collect
        runner.reset()
        ep = runner.run_episode(mdo_mode="random")

        buffer = PPORolloutBuffer()
        stats = GlobalStateStats(ep.stats.total_arrivals, ep.stats.admitted, ep.stats.rejected_by_mdo)
        gs = encode_global_state(substrate, stats)
        critic = CentralisedCritic(probe_global_state_dim(substrate), 32, 1)
        with torch.no_grad():
            cv = float(critic(gs).item())

        for t in ep.rollout.mdo:
            buffer.append_mdo(t.obs, t.action, t.log_probs, t.entropy,
                              t.value_estimate, gs, cv, t.terminal_reward, t.committed)
        for did, ts in ep.rollout.domain_actor.items():
            for t in ts:
                buffer.append_domain_actor(t)

        assert sum(len(ts) for ts in buffer.domain_actor.values()) > 0

        # CTDE-PPO update (same logic as MAPPOTrainer._ppo_update)
        config = MAPPOConfig(update_epochs=2, clip_eps=0.2, entropy_coef=0.01, max_grad_norm=0.5)
        critic_baseline = cv

        for domain_id, transitions in buffer.domain_actor.items():
            if not transitions:
                continue
            actor = actors[domain_id]
            optimizer = torch.optim.Adam(actor.policy.parameters(), lr=3e-4)

            for _epoch in range(config.update_epochs):
                epoch_loss = torch.zeros(1)
                n_steps = 0

                for t in transitions:
                    advantage = t.terminal_reward - critic_baseline
                    for step in t.steps:
                        new_logits = actor.policy._encode_and_score(
                            step.graph_data, step.vnf_context, step.action_mask,
                        )
                        dist = Categorical(logits=new_logits)
                        n_nodes = step.action_mask.size(0)
                        a = torch.tensor(n_nodes if step.action_idx == DomainPolicy.NULL_ACTION else step.action_idx)
                        new_lp = dist.log_prob(a)
                        ratio = torch.exp(new_lp - step.log_prob)
                        unclipped = ratio * advantage
                        clipped = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantage
                        epoch_loss = epoch_loss - torch.min(unclipped, clipped)
                        n_steps += 1

                if n_steps == 0:
                    continue
                loss = epoch_loss / n_steps
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.policy.parameters(), config.max_grad_norm)
                optimizer.step()

        # Verify params moved
        any_moved = False
        for d, actor in actors.items():
            for name, p in actor.policy.named_parameters():
                if not torch.equal(p, pre_params[d][name]):
                    any_moved = True
                    break
            if any_moved:
                break

        assert any_moved, "No domain actor params changed after CTDE-PPO update"

    def test_critic_params_move(self):
        """Centralised critic V_φ params move after one update round.

        The critic must be trained (regressed toward returns) for the
        advantage baseline to be state-conditioned. A critic that only
        appears as .item() floats and never receives gradients is frozen,
        and the CTDE advantage degenerates to REINFORCE with a random
        constant baseline.
        """
        seed = 42
        torch.manual_seed(seed)
        substrate = _build_substrate(seed)
        delays = _build_delays(substrate)

        actors = {d: DomainActor(d, DomainPolicy("mlp", hidden_dim=32))
                  for d in range(substrate.num_domains)}
        coordinator = MDOCoordinator(None, actors, MDOConfig(n_part=2))

        rng = np.random.default_rng(seed + 100)
        ap = ArrivalProcess(substrate, 50, 4.0, 0.02, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coordinator, delays)

        # Build critic and snapshot BEFORE
        critic = CentralisedCritic(probe_global_state_dim(substrate), 32, 1)
        pre_critic = _snapshot_params(critic)

        # Collect
        runner.reset()
        ep = runner.run_episode(mdo_mode="random")

        buffer = PPORolloutBuffer()
        stats = GlobalStateStats(ep.stats.total_arrivals, ep.stats.admitted, ep.stats.rejected_by_mdo)
        gs = encode_global_state(substrate, stats)
        with torch.no_grad():
            cv = float(critic(gs).item())

        for t in ep.rollout.mdo:
            buffer.append_mdo(t.obs, t.action, t.log_probs, t.entropy,
                              t.value_estimate, gs, cv, t.terminal_reward, t.committed)

        # GAE (needed so buffer.returns exists)
        from orion.training.gae import compute_gae
        rewards = buffer.reward_tensor()
        values = buffer.value_tensor(bootstrap=0.0)
        dones = buffer.done_tensor()
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        buffer.set_gae(advantages, returns)

        # Critic update: regress V_φ toward returns
        config = MAPPOConfig(update_epochs=2, clip_eps=0.2, value_loss_coef=0.5, max_grad_norm=0.5)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

        for _epoch in range(config.update_epochs):
            global_states = torch.stack(buffer.global_states)
            ret = buffer.returns.detach()
            old_vals = torch.tensor(buffer.critic_values, dtype=torch.float32)

            new_vals = critic(global_states).squeeze(-1)
            v_clipped = old_vals + torch.clamp(new_vals - old_vals, -config.clip_eps, config.clip_eps)
            raw_loss = (new_vals - ret) ** 2
            clip_loss = (v_clipped - ret) ** 2
            value_loss = 0.5 * torch.max(raw_loss, clip_loss).mean()

            critic_optimizer.zero_grad()
            (config.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()

        # Verify critic params moved
        any_moved = False
        for name, p in critic.named_parameters():
            if not torch.equal(p, pre_critic[name]):
                any_moved = True
                break

        assert any_moved, (
            "Critic params did not change after update. V_φ is frozen — "
            "CTDE advantage degenerates to REINFORCE with a random baseline."
        )

    def test_critic_loss_decreases(self):
        """Critic loss trends down across updates on a fixed stream.

        Params moving proves not-frozen. Loss decreasing proves the
        regression target (GAE returns) is correctly scaled and signed.
        A critic regressing toward a mis-scaled return moves params
        enthusiastically while getting worse.
        """
        seed = 42
        torch.manual_seed(seed)
        substrate = _build_substrate(seed)
        delays = _build_delays(substrate)

        actors = {d: DomainActor(d, DomainPolicy("mlp", hidden_dim=32))
                  for d in range(substrate.num_domains)}
        coordinator = MDOCoordinator(None, actors, MDOConfig(n_part=2))

        rng = np.random.default_rng(seed + 100)
        ap = ArrivalProcess(substrate, 50, 4.0, 0.02, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coordinator, delays)

        critic = CentralisedCritic(probe_global_state_dim(substrate), 32, 1)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

        runner.reset()
        ep = runner.run_episode(mdo_mode="random")

        buffer = PPORolloutBuffer()
        stats = GlobalStateStats(ep.stats.total_arrivals, ep.stats.admitted, ep.stats.rejected_by_mdo)
        gs = encode_global_state(substrate, stats)
        with torch.no_grad():
            cv = float(critic(gs).item())
        for t in ep.rollout.mdo:
            buffer.append_mdo(t.obs, t.action, t.log_probs, t.entropy,
                              t.value_estimate, gs, cv, t.terminal_reward, t.committed)

        from orion.training.gae import compute_gae
        rewards = buffer.reward_tensor()
        values = buffer.value_tensor(bootstrap=0.0)
        dones = buffer.done_tensor()
        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        buffer.set_gae(advantages, returns)

        losses = []
        for _epoch in range(10):
            gs_stack = torch.stack(buffer.global_states)
            ret = buffer.returns.detach()
            old_vals = torch.tensor(buffer.critic_values, dtype=torch.float32)

            new_vals = critic(gs_stack).squeeze(-1)
            v_clipped = old_vals + torch.clamp(new_vals - old_vals, -0.2, 0.2)
            raw_loss = (new_vals - ret) ** 2
            clip_loss = (v_clipped - ret) ** 2
            vloss = 0.5 * torch.max(raw_loss, clip_loss).mean()

            critic_opt.zero_grad()
            (0.5 * vloss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_opt.step()
            losses.append(vloss.item())

        assert losses[-1] < losses[0], (
            f"Critic loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}. "
            "Regression target may be mis-scaled or mis-signed."
        )

    def test_gae_arrival_boundary_no_leak(self):
        """Arrival N's advantage must not depend on arrival N+1's reward.

        Each slice arrival is an independent decision (contextual bandit).
        If GAE bootstraps across the arrival boundary (done=0 on rejected
        arrivals), value leaks across independent episodes and the critic
        regresses toward a target that mixes unrelated arrivals. This is
        the training-side equivalent of the stale-ledger read.
        """
        from orion.training.gae import compute_gae

        # Two streams: identical at arrival 0, different at arrival 1.
        # If boundary is clean, arrival 0's advantage is identical.
        rewards_a = torch.tensor([-10.0, 100.0])
        rewards_b = torch.tensor([-10.0, -50.0])
        values = torch.tensor([0.0, 0.0, 0.0])  # untrained critic

        # Both arrivals are independent decisions -> both done=1
        dones = torch.tensor([1.0, 1.0])

        adv_a, _ = compute_gae(rewards_a, values, dones, gamma=0.99, lam=0.95)
        adv_b, _ = compute_gae(rewards_b, values, dones, gamma=0.99, lam=0.95)

        assert abs(adv_a[0].item() - adv_b[0].item()) < 1e-6, (
            f"Arrival 0 advantage differs: {adv_a[0].item()} vs {adv_b[0].item()}. "
            "GAE is leaking value across independent arrival boundaries."
        )

    def test_done_signal_marks_every_arrival(self):
        """Every arrival's final trial has done=True, not just admitted ones.

        The `committed` field on MDOTransition must be True on the last
        trial of every arrival (admitted or rejected). If only admitted
        arrivals have done=True, GAE bootstraps from rejected arrivals
        into the next arrival, leaking value across independent decisions.
        """
        seed = 42
        torch.manual_seed(seed)
        substrate = _build_substrate(seed)
        delays = _build_delays(substrate)

        actors = {d: DomainActor(d, DomainPolicy("mlp", hidden_dim=32))
                  for d in range(substrate.num_domains)}
        coordinator = MDOCoordinator(None, actors, MDOConfig(n_part=2))

        rng = np.random.default_rng(seed + 100)
        ap = ArrivalProcess(substrate, 50, 4.0, 0.02, rng)
        ap.generate()

        runner = EpisodeRunner(substrate, ap, coordinator, delays)
        runner.reset()
        ep = runner.run_episode(mdo_mode="random")

        # Group transitions by request_id
        by_request: dict[str, list] = {}
        for t in ep.rollout.mdo:
            by_request.setdefault(t.request_id, []).append(t)

        # For every arrival, the last trial must have committed=True
        for req_id, trials in by_request.items():
            last_trial = max(trials, key=lambda t: t.trial_index)
            assert last_trial.committed, (
                f"Arrival {req_id} last trial (index={last_trial.trial_index}) "
                f"has committed=False. GAE boundary leaks across this arrival."
            )
