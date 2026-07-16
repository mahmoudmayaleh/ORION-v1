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


class TestGAEConformanceToy:
    """PREREG §N.1 / SCOPE_N C.1 — the toy that gates the conformance-fix merge.

    Two assertions (ruling 1): a mechanics check, and — the one that actually
    justifies the fix — a counterfactual showing a downstream reject *lowers*
    the earlier decision's advantage. This is 'hotspot-now costs-you-later' in
    test form, and it is what a per-arrival bandit cannot represent.
    """

    GAMMA = 0.99
    LAM = 0.95
    VALUES = torch.tensor([4.0, 1.0, 4.0, 0.0])  # per-step V + bootstrap tail 0
    DONES = torch.tensor([0.0, 0.0, 1.0])  # single episode (stream), done only at the tail

    def test_mechanics_single_stream(self) -> None:
        # rewards = [admit, reject, admit]; values/dones as above.
        rewards = torch.tensor([5.0, 0.0, 5.0])
        adv, ret = compute_gae(rewards, self.VALUES, self.DONES, self.GAMMA, self.LAM)
        # Hand-computed (see SCOPE_N C.1):
        assert adv[0] == pytest.approx(5.6584, abs=1e-3)
        assert adv[1] == pytest.approx(3.9005, abs=1e-3)
        assert adv[2] == pytest.approx(1.0, abs=1e-6)
        assert ret[0] == pytest.approx(9.6584, abs=1e-3)
        # Contrast the OLD per-arrival bandit on the same inputs: adv = r - V.
        bandit_adv = rewards - self.VALUES[:-1]
        assert torch.allclose(bandit_adv, torch.tensor([1.0, -1.0, 1.0]))
        # The pipelines differ — GAE couples across steps, the bandit does not.
        assert not torch.allclose(adv, bandit_adv)

    def test_counterfactual_reject_strictly_lowers_prior_advantage(self) -> None:
        # Identical at t=0 and t=2; flip ONLY t=1 admit(5) -> reject(0).
        stream_admit = torch.tensor([5.0, 5.0, 5.0])
        stream_reject = torch.tensor([5.0, 0.0, 5.0])
        adv_admit, _ = compute_gae(stream_admit, self.VALUES, self.DONES, self.GAMMA, self.LAM)
        adv_reject, _ = compute_gae(stream_reject, self.VALUES, self.DONES, self.GAMMA, self.LAM)
        # Arrival 0's advantage is STRICTLY lower when t=1 rejects...
        assert adv_reject[0] < adv_admit[0]
        # ...by exactly the γλ-propagated reward gap (one hop from t=0): (γλ)^1 · Δr₁.
        delta_r1 = 5.0 - 0.0
        assert (adv_admit[0] - adv_reject[0]).item() == pytest.approx(
            self.GAMMA * self.LAM * delta_r1
        )
