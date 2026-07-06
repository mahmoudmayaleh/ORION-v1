"""β_t schedule tests — pin the linear-decay contract and ablation behaviour."""

from __future__ import annotations

import pytest

from orion.training.kl_schedule import beta_constant, beta_linear, beta_zero


class TestZeroAndConstant:
    def test_zero_always_zero(self) -> None:
        assert beta_zero(0) == 0.0
        assert beta_zero(10_000) == 0.0

    def test_constant_does_not_decay(self) -> None:
        assert beta_constant(0, 0.5) == 0.5
        assert beta_constant(10_000, 0.5) == 0.5


class TestLinearDecay:
    @pytest.mark.parametrize("decay_steps", [100, 10_000, 500_000])
    def test_initial_value_at_step_zero(self, decay_steps: int) -> None:
        assert beta_linear(0, 1.0, 0.0, decay_steps) == 1.0

    @pytest.mark.parametrize("decay_steps", [100, 10_000, 500_000])
    def test_final_value_at_decay_end(self, decay_steps: int) -> None:
        assert beta_linear(decay_steps, 1.0, 0.0, decay_steps) == 0.0
        # Beyond decay_steps, stay pinned at final.
        assert beta_linear(decay_steps * 10, 1.0, 0.0, decay_steps) == 0.0

    def test_midpoint_is_arithmetic_mean(self) -> None:
        assert beta_linear(50, 1.0, 0.0, 100) == pytest.approx(0.5)
        assert beta_linear(50, 2.0, 1.0, 100) == pytest.approx(1.5)

    def test_monotone_decay(self) -> None:
        # Strictly non-increasing across the decay window.
        prev = beta_linear(0, 1.0, 0.0, 1000)
        for step in range(100, 1000, 100):
            cur = beta_linear(step, 1.0, 0.0, 1000)
            assert cur <= prev
            prev = cur

    def test_decay_steps_zero_returns_final_immediately(self) -> None:
        # Guard against divide-by-zero in pathological config.
        assert beta_linear(0, 1.0, 0.0, 0) == 0.0
