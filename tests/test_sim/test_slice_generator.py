"""Slice generator coupling tests.

The slice generator must derive every bandwidth quantity from a single β_in
draw (v4 Eq. 3). The previous independent `throughput` draw is gone — these
tests pin the new contract so it can't silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from orion.config import TopologyConfig
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import SliceType


@pytest.fixture
def substrate() -> SubstrateNetwork:
    rng = np.random.default_rng(42)
    return generate_multi_domain_topology(
        TopologyConfig(
            num_domains=3,
            nodes_per_domain=[5, 5, 5],
            intra_link_density=0.6,
            inter_domain_links=2,
        ),
        rng,
    )


class TestBetaInDerivation:
    @pytest.mark.parametrize("slice_type", list(SliceType))
    def test_min_throughput_equals_first_flow_demand_divided_by_first_vcr(
        self, substrate, slice_type
    ) -> None:
        """β_{1,2} = β_in · ρ_1  ⇒  β_in = β_{1,2} / ρ_1.

        Confirms min_throughput (= β_in) is consistent with the first flow's
        derived bandwidth_demand.
        """
        rng = np.random.default_rng(123)
        for _ in range(20):  # 20 trials per slice_type
            req = generate_slice_request("req_x", substrate, rng, slice_type=slice_type)
            if not req.flow_edges:
                continue
            # round(_, 1) in the generator + division by VCR can stack
            # ±0.2 absolute on small slice types (e.g., mMTC β_in ∈ [1, 10]).
            recovered_beta_in = req.flow_edges[0].bandwidth_demand / req.vnfs[0].vcr
            assert recovered_beta_in == pytest.approx(req.qos.min_throughput, abs=0.25)

    @pytest.mark.parametrize("slice_type", list(SliceType))
    def test_every_flow_derives_from_same_beta_in(self, substrate, slice_type) -> None:
        """β_{k,k+1} = β_in · ∏_{j=1}^{k+1} ρ_j must hold for every k."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            req = generate_slice_request("req_y", substrate, rng, slice_type=slice_type)
            beta_in = req.qos.min_throughput
            cumulative_vcr = 1.0
            for k, flow in enumerate(req.flow_edges):
                cumulative_vcr *= req.vnfs[k].vcr
                expected = beta_in * cumulative_vcr
                # Tolerate the round(_, 1) inside the generator.
                assert flow.bandwidth_demand == pytest.approx(expected, abs=0.15)

    def test_no_independent_throughput_field_in_qos_profiles(self) -> None:
        """The independent `throughput` draw is gone — guard against
        someone adding it back without thinking."""
        from orion.sim.slice_generator import _QOS_PROFILES

        for profile in _QOS_PROFILES.values():
            assert "throughput" not in profile, (
                "Independent `throughput` draw was reintroduced. The "
                "C5b floor IS β_in; do not draw it separately."
            )
            assert "beta_in" in profile
