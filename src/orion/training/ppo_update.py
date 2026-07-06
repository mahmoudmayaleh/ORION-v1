"""PPO update for the MDO actor + centralised critic.

Returns a metrics dict with each loss component as a *separate scalar*:

    policy_loss              clipped surrogate objective
    value_loss               critic loss (V_φ vs returns; clipped if config says)
    entropy_bonus            mean entropy of π^MDO (NOT signed into a regularisation lump)
    kl_prior_term            β_t · analytical_KL(π^MDO ‖ π_prior)
    approx_kl                ratio-based KL diagnostic for early stopping
    clip_fraction            fraction of samples where the clip activated
    total_loss               the actual back-prop target

Separate logging is a hard requirement: KL prior and entropy bonus both
shape exploration in different ways, and fusing them into a single
"regularisation" scalar makes diagnosis impossible.

Domain-actor PPO update follows the same skeleton but uses per-domain
log-prob tensors from `DomainResponse.log_probs`. The actor side is
substantively similar; this file implements the MDO side concretely and
points the actor update at the same helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from orion.mdo.kl_prior import analytical_kl, build_prior_logits


@dataclass
class PPOMetrics:
    """Per-update metrics. All scalars; the trainer logs each separately."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_bonus: float = 0.0
    kl_prior_term: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    total_loss: float = 0.0


def ppo_mdo_update(
    mdo_policy: nn.Module,
    critic: nn.Module,
    mdo_obs: torch.Tensor,           # [B, obs_dim]
    tier_masks: torch.Tensor,        # [B, K_max, M]
    actions: torch.Tensor,           # [B, K_max] (padded; -1 for unused slots)
    old_log_probs: torch.Tensor,     # [B, K_max]
    advantages: torch.Tensor,        # [B]
    returns: torch.Tensor,           # [B]
    global_states: torch.Tensor,     # [B, global_state_dim]
    old_values: torch.Tensor,        # [B] — for value clipping
    num_vnfs: list[int],             # actual K per sample
    prior_temperature: float,
    suggested_partitions: list[list[int]],  # m̃ per sample
    clip_eps: float,
    value_loss_coef: float,
    entropy_coef: float,
    kl_prior_beta: float,
    clip_value_loss: bool,
) -> tuple[torch.Tensor, PPOMetrics]:
    """One PPO minibatch update for the MDO policy and critic.

    Returns:
        (total_loss tensor for backprop, metrics).
    """
    B = mdo_obs.shape[0]
    if B == 0:
        return torch.zeros(1, requires_grad=True), PPOMetrics()

    policy_loss_sum = torch.zeros(1)
    value_loss_sum = torch.zeros(1)
    entropy_sum = 0.0
    kl_prior_sum = 0.0
    approx_kl_sum = 0.0
    clip_count = 0
    sample_count = 0

    new_values_full = critic(global_states)  # [B]

    for i in range(B):
        k = num_vnfs[i]
        if k == 0:
            continue
        # Re-evaluate the actions under the current policy.
        new_log_probs, mean_ent, new_logits = mdo_policy.evaluate_actions(
            obs=mdo_obs[i],
            tier_mask=tier_masks[i],
            actions=actions[i, :k],
            num_vnfs=k,
        )
        sample_count += 1

        # Clipped surrogate objective, summed across VNF slots.
        ratio = torch.exp(new_log_probs - old_log_probs[i, :k])
        adv = advantages[i].detach()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        per_slot_loss = -torch.min(unclipped, clipped)
        policy_loss_sum = policy_loss_sum + per_slot_loss.mean()

        clip_count += int(((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).sum().item())

        # KL diagnostic (for early stopping / monitoring).
        approx_kl_sum += float((old_log_probs[i, :k] - new_log_probs).mean().item())

        # KL-prior term (closed form against Agent B's suggestion).
        prior_logits = build_prior_logits(
            suggested_partitions[i][:k],
            num_domains=new_logits.shape[-1],
            tier_masks=tier_masks[i, :k],
            temperature=prior_temperature,
        )
        kl_prior = analytical_kl(new_logits, prior_logits, tier_masks[i, :k])
        kl_prior_sum += float(kl_prior.item())

        entropy_sum += float(mean_ent.item() if hasattr(mean_ent, "item") else mean_ent)

    if sample_count == 0:
        return torch.zeros(1, requires_grad=True), PPOMetrics()

    policy_loss = policy_loss_sum / sample_count

    # Critic loss with optional clipping (CleanRL detail).
    new_values = new_values_full
    v_clipped = old_values + torch.clamp(
        new_values - old_values, -clip_eps, clip_eps
    )
    raw_loss = (new_values - returns) ** 2
    clip_loss = (v_clipped - returns) ** 2
    value_loss = (
        0.5 * torch.max(raw_loss, clip_loss).mean()
        if clip_value_loss
        else 0.5 * raw_loss.mean()
    )

    entropy_mean = entropy_sum / sample_count
    kl_prior_mean = kl_prior_sum / sample_count
    approx_kl_mean = approx_kl_sum / sample_count
    clip_frac = clip_count / max(sample_count, 1)

    total = (
        policy_loss
        + value_loss_coef * value_loss
        - entropy_coef * entropy_mean
        + kl_prior_beta * kl_prior_mean
    )

    metrics = PPOMetrics(
        policy_loss=float(policy_loss.item()),
        value_loss=float(value_loss.item()),
        entropy_bonus=float(entropy_coef * entropy_mean),
        kl_prior_term=float(kl_prior_beta * kl_prior_mean),
        approx_kl=approx_kl_mean,
        clip_fraction=clip_frac,
        total_loss=float(total.item()),
    )
    return total, metrics


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Diagnostic: how much of the return variance the critic captures.

    1.0 = perfect, 0.0 = no better than mean, <0 = worse than mean.
    Standard PPO debug metric.
    """
    var_y = returns.var()
    if var_y <= 1e-8:
        return 0.0
    return float(1.0 - (returns - values).var() / var_y)
