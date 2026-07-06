"""Analytical KL divergence for the MDO soft-prior regularisation.

Computes KL(π^MDO || π_prior) in closed form over factored Categorical
distributions, one per VNF slot. Both distributions are discrete and
tractable, so no Monte Carlo estimator is needed.

The analytical expression is differentiated directly by PyTorch autograd.
This avoids the gradient-estimator pitfall documented in:
  - Tang & Munos, arXiv:2506.09477, 2025
  - Huang et al., "A Comedy of Estimators", arXiv:2512.21852, 2025

Usage:
    kl = analytical_kl(logits_mdo, logits_prior, mask)
    loss = loss_ppo + beta * kl
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def analytical_kl(
    logits_mdo: torch.Tensor,
    logits_prior: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute analytical KL(π^MDO || π_prior) over factored Categoricals.

    Args:
        logits_mdo: MDO policy logits, shape [K, D_max] where K is the number
            of VNF slots and D_max is the maximum number of domains.
        logits_prior: Prior logits (from Agent B's suggested partition),
            same shape [K, D_max].
        mask: Boolean mask of valid (tier-feasible) domains per VNF,
            shape [K, D_max]. True = valid. If None, all domains are valid.
        eps: Small constant for numerical stability in log.

    Returns:
        Scalar KL divergence, safe to differentiate via autograd.

    The computation:
        KL = Σ_k Σ_{d ∈ D(τ_{f_k})} π^MDO(d|k) · log(π^MDO(d|k) / π_prior(d|k))

    Masked positions are set to -inf logit before softmax so they receive
    zero probability mass and contribute zero to the KL sum.
    """
    # Use a large negative finite value instead of -inf to avoid NaN gradients.
    # -inf in logits → 0 prob after softmax, but 0 * log(0) = NaN in autograd.
    # -1e9 gives effectively zero prob (~exp(-1e9)) with clean gradients.
    NEGINF_SAFE = torch.tensor(-1e9, dtype=logits_mdo.dtype, device=logits_mdo.device)

    if mask is not None:
        logits_mdo = torch.where(mask, logits_mdo, NEGINF_SAFE)
        logits_prior = torch.where(mask, logits_prior, NEGINF_SAFE)

        valid_per_slot = mask.any(dim=-1)  # [K]
        if not valid_per_slot.all():
            valid_logits_mdo = logits_mdo[valid_per_slot]
            valid_logits_prior = logits_prior[valid_per_slot]
            if valid_logits_mdo.numel() == 0:
                return torch.tensor(0.0, dtype=logits_mdo.dtype, device=logits_mdo.device)
            return analytical_kl(valid_logits_mdo, valid_logits_prior)

    log_p = F.log_softmax(logits_mdo, dim=-1)
    log_q = F.log_softmax(logits_prior, dim=-1)
    p = log_p.exp()

    kl_elementwise = p * (log_p - log_q)
    kl_per_slot = kl_elementwise.sum(dim=-1)  # [K]

    # Clamp to zero to handle numerical noise
    kl_per_slot = kl_per_slot.clamp(min=0.0)

    return kl_per_slot.sum()


def build_prior_logits(
    suggested_domains: list[int],
    num_domains: int,
    tier_masks: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build temperature-softened prior logits centred on Agent B's suggestion.

    Args:
        suggested_domains: Agent B's suggested domain per VNF, list of length K.
        num_domains: Total number of domains M.
        tier_masks: Boolean mask [K, M] of tier-feasible domains per VNF.
        temperature: Softening temperature. Lower = sharper peak on m̃.
            At temperature → 0, this becomes a one-hot on m̃.
            At temperature → ∞, this becomes uniform over feasible domains.

    Returns:
        Prior logits [K, M], ready for use with analytical_kl().
    """
    K = len(suggested_domains)
    logits = torch.zeros(K, num_domains)

    for k, d in enumerate(suggested_domains):
        # Place a peak on the suggested domain
        logits[k, d] = 1.0 / temperature

    # Mask infeasible domains with large negative value (analytical_kl
    # also masks, but setting here keeps prior logits self-consistent)
    logits = torch.where(tier_masks, logits, torch.tensor(-1e9))

    return logits


def beta_schedule(
    step: int,
    total_steps: int,
    beta_start: float = 1.0,
    beta_end: float = 0.01,
) -> float:
    """Linear decay schedule for the KL regularisation weight β_t.

    Args:
        step: Current training step.
        total_steps: Total training steps.
        beta_start: Initial β (high = strong LLM prior influence).
        beta_end: Final β (near-zero = MDO deviates freely).

    Returns:
        Current β_t value.
    """
    if total_steps <= 0:
        return beta_end
    frac = min(step / total_steps, 1.0)
    return beta_start + (beta_end - beta_start) * frac
