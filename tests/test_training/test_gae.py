"""GAE tests — pin numerical correctness against the closed-form recurrence."""

from __future__ import annotations

import pytest
import torch

from orion.training.gae import compute_gae


class TestGAEShape:
    def test_returns_match_shape(self) -> None:
        T = 8
        rewards = torch.zeros(T)
        values = torch.zeros(T + 1)
        dones = torch.zeros(T)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert adv.shape == rewards.shape
        assert ret.shape == rewards.shape

    def test_wrong_value_length_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_gae(
                rewards=torch.zeros(4),
                values=torch.zeros(4),  # should be 5
                dones=torch.zeros(4),
                gamma=0.99,
                lam=0.95,
            )


class TestGAECorrectness:
    def test_zero_rewards_zero_values_gives_zero_advantage(self) -> None:
        rewards = torch.zeros(5)
        values = torch.zeros(6)
        dones = torch.zeros(5)
        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95)
        assert torch.allclose(adv, torch.zeros(5))
        assert torch.allclose(ret, torch.zeros(5))

    def test_done_resets_bootstrap(self) -> None:
        # If episode ends at step T-1, the bootstrap value at T should be masked.
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.5, 2.0, 100.0])  # absurd bootstrap to detect masking
        dones = torch.tensor([0.0, 1.0])
        adv, _ = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)
        # At t=1, not_done=0 so delta_1 = r_1 - V_1 = -2.0; gae=-2.0; adv[1]=-2.0
        # At t=0, not_done=1, delta_0 = r_0 + γ·V_1·1 - V_0 = 1 + 0.9·2 - 0.5 = 2.3
        # gae = 2.3 + γ·λ·not_done(t=0)·prev_gae = 2.3 + 0.9·1·(-2.0) = 0.5
        assert adv[1] == pytest.approx(-2.0)
        assert adv[0] == pytest.approx(0.5)

    def test_lam_zero_collapses_to_td_advantage(self) -> None:
        # λ=0 → GAE reduces to one-step TD: A_t = r_t + γV_{t+1} − V_t.
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.0, 1.0, 1.0, 1.0])
        dones = torch.zeros(3)
        adv, _ = compute_gae(rewards, values, dones, gamma=0.9, lam=0.0)
        expected = torch.tensor([
            1.0 + 0.9 * 1.0 - 0.0,
            2.0 + 0.9 * 1.0 - 1.0,
            3.0 + 0.9 * 1.0 - 1.0,
        ])
        assert torch.allclose(adv, expected)
