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


class DirectJointPolicy(nn.Module):
    """Single Categorical over enumerated feasible joint partitions.

    Maximally expressive: represents the full joint π(a¹,a²,...,aᴷ) with no
    factorization. For small instances (M^K ≤ few hundred), this is the clean
    test of representation vs credit assignment.
    """

    def __init__(
        self,
        obs_dim: int,
        num_domains: int,
        max_chain_length: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.max_chain_length = max_chain_length
        self.max_joint = num_domains ** max_chain_length

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
        self.actor_head = nn.Linear(hidden_dim, self.max_joint)
        self.aux_value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)

    def _enumerate_and_mask(
        self, tier_mask: torch.Tensor, num_vnfs: int,
    ) -> tuple[list[tuple[int, ...]], torch.Tensor]:
        import itertools
        partitions = list(itertools.product(range(self.num_domains), repeat=num_vnfs))
        mask = torch.ones(len(partitions), dtype=torch.bool)
        for i, p in enumerate(partitions):
            for k, d in enumerate(p):
                if not tier_mask[k, d]:
                    mask[i] = False
                    break
        return partitions, mask

    def _assert_chain(self, num_vnfs: int) -> None:
        # RC is asserted K<=max_chain_length. A longer chain would overrun the
        # enumerated head (num_domains**max_chain_length) — stop-and-report per
        # §U.1, never a silent truncation.
        if num_vnfs > self.max_chain_length:
            raise ValueError(
                f"DirectJointPolicy: num_vnfs={num_vnfs} exceeds "
                f"max_chain_length={self.max_chain_length}; joint enumeration "
                f"head has only num_domains**max_chain_length atoms."
            )

    def forward(
        self,
        obs: torch.Tensor,
        tier_mask: torch.Tensor,
        num_vnfs: int,
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, float]:
        self._assert_chain(num_vnfs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.encoder(obs)
        raw_logits = self.actor_head(h).squeeze(0)

        partitions, mask = self._enumerate_and_mask(tier_mask, num_vnfs)
        n_joint = len(partitions)
        logits = raw_logits[:n_joint]
        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype)
        masked_logits = torch.where(mask, logits, neg_inf)

        dist = Categorical(logits=masked_logits)
        if deterministic:
            action_idx = masked_logits.argmax()
        else:
            action_idx = dist.sample()

        partition = list(partitions[action_idx.item()])
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy().item()
        return partition, log_prob.unsqueeze(0), masked_logits.unsqueeze(0), entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        tier_mask: torch.Tensor,
        actions: torch.Tensor,
        num_vnfs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_chain(num_vnfs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.encoder(obs)
        raw_logits = self.actor_head(h).squeeze(0)

        partitions, mask = self._enumerate_and_mask(tier_mask, num_vnfs)
        n_joint = len(partitions)
        logits = raw_logits[:n_joint]
        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype)
        masked_logits = torch.where(mask, logits, neg_inf)

        target = tuple(actions.tolist())
        action_idx = torch.tensor(partitions.index(target), dtype=torch.long)
        dist = Categorical(logits=masked_logits)
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        return log_prob.unsqueeze(0), entropy, masked_logits.unsqueeze(0)

    def joint_prior_logits(
        self,
        tier_mask: torch.Tensor,
        num_vnfs: int,
        suggested_canonical: list[int],
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Single-atom joint soft-prior over enumerated feasible joints (§U.1b).

        Peaks logit 1/temperature on the ONE joint atom equal to the suggested
        partition m̃ (canonical frame), 0 on the other feasible atoms, infeasible
        atoms left to be masked by analytical_kl. Atom ordering matches
        `evaluate_actions` (same `_enumerate_and_mask`).

        Returns (prior_logits[1, n_joint], joint_mask[1, n_joint]) ready for
        `analytical_kl(new_logits, prior_logits, joint_mask)`, or None if m̃ is
        not a feasible atom (caller skips the KL term — never fakes it).
        """
        self._assert_chain(num_vnfs)
        partitions, mask = self._enumerate_and_mask(tier_mask, num_vnfs)
        target = tuple(int(d) for d in suggested_canonical)
        try:
            idx = partitions.index(target)
        except ValueError:
            return None
        if not bool(mask[idx]):
            return None  # suggested partition is tier-infeasible under this mask
        peak = 1.0 / max(float(temperature), 1e-6)
        logits = torch.zeros(len(partitions), dtype=torch.float32)
        logits[idx] = peak
        return logits.unsqueeze(0), mask.unsqueeze(0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.encoder(obs)
        return self.aux_value_head(h).squeeze(-1)


class AutoregMDOPolicy(nn.Module):
    """Autoregressive sequential-decode MDO policy (§U.1e).

    Decodes per-VNF domain assignments in SFC order, each step CONDITIONED on
    where prior VNFs of this slice were placed (a per-domain running count). This
    captures the joint correlation the factored MDOPolicy could not express
    (colocation) WITHOUT enumerating the M^K joint space, so it scales to RC's
    K up to 6+ where DirectJointPolicy (5^6 = 15625 atoms) is infeasible. No
    chain-length cap.

    Interface matches MDOPolicy exactly (per-VNF [K, M] raw logits + [K] log-probs),
    so it drops into the factored KL/PPO path — but the log-probs are CONDITIONAL
    (their product is the joint) and the KL toward m̃ is per-step.
    """

    def __init__(self, obs_dim, num_domains, max_vnfs=10, hidden_dim=128, num_layers=2):
        super().__init__()
        self.num_domains = num_domains
        self.max_vnfs = max_vnfs
        self.hidden_dim = hidden_dim
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        # Decoder input: [h | per-domain running-count frac (M) | position frac (1)].
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + num_domains + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_domains))
        self.aux_value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)

    def _encode(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.encoder(obs).squeeze(0)  # [hidden]

    def _step_logits(self, h, counts, k, num_vnfs):
        cnt = torch.tensor(counts, dtype=h.dtype, device=h.device) / max(num_vnfs, 1)
        pos = torch.tensor([k / max(num_vnfs, 1)], dtype=h.dtype, device=h.device)
        return self.decoder(torch.cat([h, cnt, pos]))  # [M] raw (pre-mask)

    def forward(self, obs, tier_mask, num_vnfs, deterministic=False,
                prior_logits=None, prior_weight=0.0):
        """prior_logits [K, M] + prior_weight>0 make the LLM plan ADVISE the decode
        at inference: the suggested-domain bias is added to the DECISION logits so
        m~ always steers the committed partition. raw_list keeps UNBIASED logits."""
        h = self._encode(obs)
        neg_inf = torch.tensor(float("-inf"), dtype=h.dtype, device=h.device)
        counts = [0.0] * self.num_domains
        partition, lp_list, raw_list = [], [], []
        ent_sum = 0.0
        for k in range(num_vnfs):
            raw = self._step_logits(h, counts, k, num_vnfs)
            dec = raw if prior_logits is None else raw + prior_weight * prior_logits[k]
            masked = torch.where(tier_mask[k], dec, neg_inf)
            dist = Categorical(logits=masked)
            a = masked.argmax() if deterministic else dist.sample()
            ai = int(a.item())
            partition.append(ai)
            lp_list.append(dist.log_prob(a))
            ent_sum += dist.entropy().item()
            raw_list.append(raw)
            counts[ai] += 1.0
        log_probs = torch.stack(lp_list)
        logits = torch.stack(raw_list)  # [K, M] raw
        mean_ent = ent_sum / num_vnfs if num_vnfs > 0 else 0.0
        return partition, log_probs, logits, mean_ent

    def evaluate_actions(self, obs, tier_mask, actions, num_vnfs):
        h = self._encode(obs)
        neg_inf = torch.tensor(float("-inf"), dtype=h.dtype, device=h.device)
        counts = [0.0] * self.num_domains
        lp_list, raw_list = [], []
        ent_sum = torch.tensor(0.0)
        for k in range(num_vnfs):
            raw = self._step_logits(h, counts, k, num_vnfs)
            masked = torch.where(tier_mask[k], raw, neg_inf)
            dist = Categorical(logits=masked)
            lp_list.append(dist.log_prob(actions[k]))
            ent_sum = ent_sum + dist.entropy()
            raw_list.append(raw)
            counts[int(actions[k].item())] += 1.0
        log_probs = torch.stack(lp_list)
        logits = torch.stack(raw_list)  # [K, M] raw
        ent = ent_sum / num_vnfs if num_vnfs > 0 else ent_sum
        return log_probs, ent, logits

    def get_value(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.aux_value_head(self.encoder(obs)).squeeze(-1)


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
