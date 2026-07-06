"""Centralised critic V_φ(s_t) — flat MLP (Choice A1).

Training-only network: takes the global state encoded by `global_state.py`,
returns a scalar value estimate, used for advantage computation in MAPPO.
Discarded at inference — the deployment uses only the MDO's auxiliary
value head V^MDO_ψ for rejection trigger (iii).

Architecture is intentionally simple: a wide-then-narrow MLP with
LayerNorm and ReLU, orthogonal init (gain=1.0 for hidden, gain=1.0 for
the output value head — CleanRL convention). The graph structure the
actors see is deliberately not modelled here; the canonical domain
ordering already gives permutation invariance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CentralisedCritic(nn.Module):
    """V_φ(s_t) for MAPPO advantage estimation.

    Args:
        input_dim: dimension of the flat global-state tensor (probe with
            `training.global_state.probe_global_state_dim`).
        hidden_dim: width of each hidden layer.
        num_layers: number of hidden layers (default 3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init, CleanRL gains: √2 on hidden, 1.0 on value head."""
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)  # type: ignore[arg-type]
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)  # type: ignore[arg-type]
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return V_φ(s_t). Accepts [input_dim] or [B, input_dim].

        Always returns a scalar (or [B]); the trailing value-head dim is squeezed.
        """
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        h = self.encoder(global_state)
        v = self.value_head(h).squeeze(-1)
        return v.squeeze(0) if squeeze_batch else v
