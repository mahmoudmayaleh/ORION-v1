"""PPO-ready rollout buffer.

Extends `sim/rollout_buffer.MultiAgentRollout` with the per-step tensor
structures PPO needs: stored observations, advantage slots, and minibatch
iteration. Tensors live on CPU until the update step moves them to GPU.

Shape note: the MDO action is variable-length (one slot per VNF). To keep
the buffer simple, we store actions and per-slot log-probs as Python
lists/ragged structures; the PPO update flattens them per arrival. The
critic input s_t is fixed-length (probed once at start), so V_φ targets
live in a regular tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch

from orion.sim.rollout_buffer import DomainActorTransition, MDOTransition


@dataclass
class PPORolloutBuffer:
    """Append-only buffer over one rollout round.

    Per-arrival storage (lists, one element per MDO transition):
        mdo_obs       — MDO observation tensor at decision time, [obs_dim]
        mdo_action    — per-VNF domain assignment, list[int]
        mdo_log_prob  — per-VNF log-prob, [K_s]
        mdo_entropy   — scalar
        mdo_value     — V^MDO_ψ at decision time, scalar (aux head; NOT V_φ)
        global_state  — s_t for V_φ, [global_state_dim]
        critic_value  — V_φ(s_t), scalar
        reward        — terminal reward (broadcast across retries)
        done          — 1.0 if this transition ended the arrival

    All fields are populated by the trainer before/after EpisodeRunner.run.
    GAE slots (advantages, returns) are filled in by the PPO update.
    """

    mdo_obs: list[torch.Tensor] = field(default_factory=list)
    mdo_actions: list[list[int]] = field(default_factory=list)
    mdo_log_probs: list[torch.Tensor] = field(default_factory=list)
    mdo_entropies: list[float] = field(default_factory=list)
    mdo_aux_values: list[float] = field(default_factory=list)
    mdo_tier_masks: list[torch.Tensor] = field(default_factory=list)
    mdo_num_vnfs: list[int] = field(default_factory=list)
    global_states: list[torch.Tensor] = field(default_factory=list)
    critic_values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)

    # GAE outputs — populated by `set_gae`.
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    # Per-agent transitions for the actor PPO update.
    domain_actor: dict[int, list[DomainActorTransition]] = field(default_factory=dict)

    # ── Appenders ───────────────────────────────────────────────────────

    def append_mdo(
        self,
        mdo_obs: torch.Tensor,
        action: list[int],
        log_prob: torch.Tensor,
        entropy: float,
        aux_value: float,
        global_state: torch.Tensor,
        critic_value: float,
        reward: float,
        done: bool,
        tier_mask: torch.Tensor | None = None,
        num_vnfs: int = 0,
    ) -> None:
        self.mdo_obs.append(mdo_obs.detach())
        self.mdo_actions.append(action)
        self.mdo_log_probs.append(log_prob.detach())
        self.mdo_entropies.append(entropy)
        self.mdo_aux_values.append(aux_value)
        if tier_mask is not None:
            self.mdo_tier_masks.append(tier_mask.detach())
        self.mdo_num_vnfs.append(num_vnfs)
        self.global_states.append(global_state.detach())
        self.critic_values.append(critic_value)
        self.rewards.append(reward)
        self.dones.append(1.0 if done else 0.0)

    def append_domain_actor(self, transition: DomainActorTransition) -> None:
        self.domain_actor.setdefault(transition.domain_id, []).append(transition)

    # ── Stacking + GAE ──────────────────────────────────────────────────

    def reward_tensor(self) -> torch.Tensor:
        return torch.tensor(self.rewards, dtype=torch.float32)

    def value_tensor(self, bootstrap: float = 0.0) -> torch.Tensor:
        """[T+1] critic values, with `bootstrap` appended at the tail."""
        return torch.tensor(
            self.critic_values + [bootstrap], dtype=torch.float32
        )

    def done_tensor(self) -> torch.Tensor:
        return torch.tensor(self.dones, dtype=torch.float32)

    def set_gae(self, advantages: torch.Tensor, returns: torch.Tensor) -> None:
        self.advantages = advantages
        self.returns = returns

    # ── Minibatch iteration ─────────────────────────────────────────────

    def minibatches(self, minibatch_size: int) -> Iterator[list[int]]:
        """Yield random index permutations of size `minibatch_size`."""
        T = len(self.rewards)
        if T == 0:
            return
        indices = torch.randperm(T).tolist()
        for start in range(0, T, minibatch_size):
            yield indices[start : start + minibatch_size]

    # ── Bookkeeping ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.mdo_obs.clear()
        self.mdo_actions.clear()
        self.mdo_log_probs.clear()
        self.mdo_entropies.clear()
        self.mdo_aux_values.clear()
        self.mdo_tier_masks.clear()
        self.mdo_num_vnfs.clear()
        self.global_states.clear()
        self.critic_values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
        self.domain_actor.clear()


# Re-export so callers don't need a separate import.
__all__ = ["PPORolloutBuffer", "MDOTransition", "DomainActorTransition"]
