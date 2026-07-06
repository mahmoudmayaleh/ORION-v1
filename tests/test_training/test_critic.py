"""CentralisedCritic tests."""

from __future__ import annotations

import pytest
import torch

from orion.training.critic import CentralisedCritic


class TestCritic:
    def test_forward_unbatched_returns_scalar(self) -> None:
        critic = CentralisedCritic(input_dim=32, hidden_dim=64, num_layers=2)
        v = critic(torch.zeros(32))
        assert v.dim() == 0  # scalar

    def test_forward_batched_returns_vector(self) -> None:
        critic = CentralisedCritic(input_dim=32, hidden_dim=64, num_layers=2)
        v = critic(torch.zeros(4, 32))
        assert v.shape == (4,)

    def test_gradients_flow(self) -> None:
        critic = CentralisedCritic(input_dim=8, hidden_dim=16, num_layers=2)
        x = torch.randn(2, 8)
        target = torch.tensor([1.0, -1.0])
        v = critic(x)
        loss = (v - target).pow(2).mean()
        loss.backward()
        # At least one parameter should have a non-zero grad.
        non_zero = any(
            p.grad is not None and torch.abs(p.grad).sum().item() > 0
            for p in critic.parameters()
        )
        assert non_zero
