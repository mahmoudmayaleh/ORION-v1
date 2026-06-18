"""MDO policy network: factored MaskedCategorical + auxiliary value head.

The MDO outputs a per-VNF domain assignment from factored independent
Categorical distributions, one per VNF slot. The only hard mask is
tier feasibility.

Two heads share the MLP encoder:
  1. Actor head: per-VNF logits -> MaskedCategorical -> partition, log_probs, entropy
  2. Auxiliary value head V^MDO_ψ(o^MDO_t, π̃_t): scalar value estimate that
     survives at inference for rejection trigger (iii). Conceptually distinct
     from the centralised critic V_φ(s_t) which takes global state and is
     discarded at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MDOPolicy(nn.Module):
    """Factored MaskedCategorical policy + auxiliary value head.

    Args:
        obs_dim: Dimension of the flat MDO observation tensor.
        num_domains: Maximum number of domains M.
        max_vnfs: Maximum number of VNFs per slice (for output head sizing).
        hidden_dim: Hidden layer width.
        num_layers: Number of hidden layers in the shared encoder.
    """

    def __init__(
        self,
        obs_dim: int,
        num_domains: int,
        max_vnfs: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.max_vnfs = max_vnfs

        # Shared encoder
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Actor head: produces per-VNF logits over domains
        # Output shape: [max_vnfs * num_domains], reshaped to [max_vnfs, num_domains]
        self.actor_head = nn.Linear(hidden_dim, max_vnfs * num_domains)

        # Auxiliary value head V^MDO_ψ (survives inference, used by trigger iii)
        self.aux_value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization following PPO best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)  # type: ignore[arg-type]
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Larger gain for encoder layers
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)  # type: ignore[arg-type]

    def forward(
        self,
        obs: torch.Tensor,
        tier_mask: torch.Tensor,
        num_vnfs: int,
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, float]:
        """Sample a partition from the factored policy.

        Args:
            obs: Flat observation tensor, shape [obs_dim] or [B, obs_dim].
            tier_mask: Boolean mask [K, M] of tier-feasible domains per VNF.
                True = feasible. K = num_vnfs, M = num_domains.
            num_vnfs: Actual number of VNFs in this slice (K ≤ max_vnfs).
            deterministic: If True, use argmax instead of sampling.

        Returns:
            partition: Per-VNF domain assignment, list of length K.
            log_probs: Log probability per VNF slot, shape [K].
            logits: Raw logits [K, M] (for KL computation).
            entropy: Mean entropy across VNF slots (scalar).
        """
        # Handle unbatched input
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Shared encoder
        h = self.encoder(obs)  # [1, hidden_dim]

        # Actor head -> [1, max_vnfs * num_domains] -> [max_vnfs, num_domains]
        raw_logits = self.actor_head(h).view(-1, self.max_vnfs, self.num_domains)
        raw_logits = raw_logits.squeeze(0)  # [max_vnfs, num_domains]

        # Slice to actual number of VNFs
        logits = raw_logits[:num_vnfs]  # [K, M]

        # Apply tier-feasibility mask: set infeasible to -inf
        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(tier_mask[:num_vnfs], logits, neg_inf)

        # Factored independent Categorical per VNF slot
        partition = []
        log_prob_list = []
        entropy_sum = 0.0

        for k in range(num_vnfs):
            dist = Categorical(logits=masked_logits[k])

            if deterministic:
                action = masked_logits[k].argmax()
            else:
                action = dist.sample()

            partition.append(action.item())
            log_prob_list.append(dist.log_prob(action))
            entropy_sum += dist.entropy().item()

        log_probs = torch.stack(log_prob_list)
        mean_entropy = entropy_sum / num_vnfs if num_vnfs > 0 else 0.0

        return partition, log_probs, logits[:num_vnfs], mean_entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        tier_mask: torch.Tensor,
        actions: torch.Tensor,
        num_vnfs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_probs and entropy for given actions (for PPO update).

        Args:
            obs: Flat observation tensor [obs_dim].
            tier_mask: Boolean mask [K, M].
            actions: Taken actions [K] (domain indices).
            num_vnfs: Actual number of VNFs.

        Returns:
            log_probs: [K] log probabilities of the actions.
            entropy: Mean entropy (scalar tensor).
            logits: [K, M] raw logits for KL computation.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.encoder(obs)
        raw_logits = self.actor_head(h).view(-1, self.max_vnfs, self.num_domains)
        raw_logits = raw_logits.squeeze(0)
        logits = raw_logits[:num_vnfs]

        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        masked_logits = torch.where(tier_mask[:num_vnfs], logits, neg_inf)

        log_probs = []
        entropy_sum = torch.tensor(0.0)

        for k in range(num_vnfs):
            dist = Categorical(logits=masked_logits[k])
            log_probs.append(dist.log_prob(actions[k]))
            entropy_sum = entropy_sum + dist.entropy()

        return (
            torch.stack(log_probs),
            entropy_sum / num_vnfs if num_vnfs > 0 else entropy_sum,
            logits,
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary value estimate V^MDO_ψ(o^MDO_t, π̃_t).

        This is NOT the centralised critic. It survives inference and is
        used by rejection trigger (iii).

        Args:
            obs: Flat observation tensor [obs_dim] or [B, obs_dim].

        Returns:
            Scalar value estimate.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.encoder(obs)
        return self.aux_value_head(h).squeeze(-1)
