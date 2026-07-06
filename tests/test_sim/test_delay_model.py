"""M/M/1 sojourn-time model tests."""

from __future__ import annotations

import math

import pytest

from orion.sim.delay_model import link_sojourn, mm1_sojourn, node_sojourn


class TestMM1Sojourn:
    def test_zero_load_reduces_to_base_plus_one_over_mu(self) -> None:
        # base * intensity + 1/μ when load=0
        result = mm1_sojourn(base_delay=2.0, intensity=1.0, service_rate=10.0, load=0.0)
        assert result == pytest.approx(2.0 + 0.1)

    def test_intensity_scales_only_the_base(self) -> None:
        result = mm1_sojourn(base_delay=1.0, intensity=3.0, service_rate=10.0, load=0.0)
        assert result == pytest.approx(3.0 + 0.1)

    def test_load_below_capacity_is_finite(self) -> None:
        result = mm1_sojourn(base_delay=1.0, intensity=1.0, service_rate=10.0, load=5.0)
        # 1.0 + 1/(10-5) = 1.2
        assert result == pytest.approx(1.2)

    def test_load_equal_to_service_rate_saturates(self) -> None:
        result = mm1_sojourn(base_delay=1.0, intensity=1.0, service_rate=10.0, load=10.0)
        assert math.isinf(result)

    def test_load_above_capacity_saturates(self) -> None:
        result = mm1_sojourn(base_delay=1.0, intensity=1.0, service_rate=10.0, load=15.0)
        assert math.isinf(result)

    def test_sojourn_monotone_in_load(self) -> None:
        prev = mm1_sojourn(2.0, 1.0, 10.0, 1.0)
        for load in (2.0, 3.0, 5.0, 8.0, 9.5):
            cur = mm1_sojourn(2.0, 1.0, 10.0, load)
            assert cur > prev
            prev = cur


class TestNodeAndLinkSojourn:
    def test_node_sojourn_uses_cpu_capacity_as_mu(self) -> None:
        # 2ms base × 1.0 intensity + 1/(16-8) = 2 + 0.125
        result = node_sojourn(
            base_processing_delay=2.0,
            intensity=1.0,
            cpu_capacity=16.0,
            cpu_used=8.0,
        )
        assert result == pytest.approx(2.125)

    def test_link_sojourn_uses_bandwidth_as_mu(self) -> None:
        # 5ms prop + 1/(1000-100)
        result = link_sojourn(
            propagation_delay=5.0,
            bandwidth_capacity=1000.0,
            bandwidth_used=100.0,
        )
        assert result == pytest.approx(5.0 + 1.0 / 900.0)

    def test_link_sojourn_intensity_is_one(self) -> None:
        # link has no intensity scaling — should equal mm1 with intensity=1
        a = link_sojourn(5.0, 1000.0, 200.0)
        b = mm1_sojourn(5.0, 1.0, 1000.0, 200.0)
        assert a == pytest.approx(b)
