"""Running return-target normalization for the centralised critic (§O.1).

Value normalization per the MAPPO reference practice (Yu et al. 2022, ref 19;
PopArt-style running statistics, van Hasselt et al. 2016): the critic is
trained on normalized targets and its outputs are denormalized wherever
values feed GAE.

Update cadence — PINNED (§O.1, ratified 2026-07-13): statistics are updated
ONCE PER ROUND from that round's return batch, BEFORE the critic epochs, and
are FROZEN within the update loop. The normalizer state rides in the §O.7
checkpoint.
"""

from __future__ import annotations

import torch


class ValueNormalizer:
    """Running mean/std over return targets (parallel-Welford accumulation).

    Not a nn.Module on purpose: it has no gradients and its update cadence is
    pinned by §O.1 — call `update(returns)` exactly once per rollout round,
    before the critic epochs.
    """

    def __init__(self, eps: float = 1e-8, min_std: float = 1e-6) -> None:
        self.count: float = 0.0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.eps = eps
        self.min_std = min_std

    # ── Statistics ─────────────────────────────────────────────────────────

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max((self.m2 / self.count) ** 0.5, self.min_std)

    def update(self, returns: torch.Tensor) -> None:
        """Fold one round's return batch into the running statistics.

        Chan et al. parallel-Welford merge: exact regardless of batch size.
        """
        x = returns.detach().float().flatten()
        n_b = float(x.numel())
        if n_b == 0:
            return
        mean_b = float(x.mean())
        m2_b = float(((x - mean_b) ** 2).sum())

        delta = mean_b - self.mean
        n = self.count + n_b
        self.mean += delta * (n_b / n)
        self.m2 += m2_b + delta * delta * (self.count * n_b / n)
        self.count = n

    # ── Transforms ─────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return x
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return x
        return x * (self.std + self.eps) + self.mean

    # ── Checkpoint (§O.7) ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "m2": self.m2,
                "eps": self.eps, "min_std": self.min_std}

    def load_state_dict(self, state: dict) -> None:
        self.count = float(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])
        self.eps = float(state.get("eps", 1e-8))
        self.min_std = float(state.get("min_std", 1e-6))
