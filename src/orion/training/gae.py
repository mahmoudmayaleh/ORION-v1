"""Generalised Advantage Estimation (Schulman et al. 2016).

Computes A_t = Σ_{l=0}^{T−t−1} (γλ)^l · δ_{t+l}, where δ_t = r_t + γV(s_{t+1}) − V(s_t).
Standard CleanRL form.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,        # [T] or [T, N_env]
    values: torch.Tensor,          # [T+1] or [T+1, N_env] — bootstrapped value at end
    dones: torch.Tensor,           # [T] or [T, N_env] — 1.0 = episode ended at step t
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards: per-step rewards.
        values: per-step value estimates plus a bootstrap value at T.
        dones: 1.0 if the episode terminated at step t (mask future bootstraps).
        gamma: discount factor.
        lam: GAE smoothing.

    Returns:
        (advantages, returns) both shape-matching `rewards`.
    """
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError(
            f"values must have shape[0] = T+1; got values={values.shape}, rewards={rewards.shape}"
        )

    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])

    # Walk backward in time, accumulating δ_t weighted by (γλ)^l.
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns
