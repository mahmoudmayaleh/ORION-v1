"""Global-state encoder tests.

Pin the contract that Choice A1 depends on: domains are consumed in the
SAME canonical order the MDO uses, so the flat concatenation is
permutation-invariant by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from orion.config import TopologyConfig
from orion.mdo.observation import build_domain_summaries
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.training.global_state import (
    GlobalStateStats,
    encode_global_state,
    probe_global_state_dim,
)


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


class TestEncoding:
    def test_returns_finite_float32(self, substrate) -> None:
        v = encode_global_state(substrate, GlobalStateStats())
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()

    def test_probe_returns_positive_integer(self, substrate) -> None:
        d = probe_global_state_dim(substrate)
        assert isinstance(d, int)
        assert d > 0

    def test_dim_stable_across_states(self, substrate) -> None:
        """The dim depends only on the substrate topology, not on
        residuals / arrivals; Phase 5 probes once and caches the size."""
        d0 = probe_global_state_dim(substrate)
        # Mutate residuals — dim must NOT change.
        for n, dn in substrate.graph.nodes(data=True):
            dn["cpu_residual"] *= 0.5
        d1 = probe_global_state_dim(substrate)
        assert d0 == d1


class TestCanonicalOrdering:
    def test_domain_block_matches_mdo_observation_order(self, substrate) -> None:
        """The critic MUST consume domains in the same canonical order
        the MDO uses. This is the load-bearing assumption of A1.
        """
        mdo_summaries = build_domain_summaries(substrate)
        # The encoder builds per-domain features in this exact iteration
        # order. Confirm by checking the first per-domain block matches
        # the first MDO domain's cpu fraction (idx 0 of each block).
        from orion.training.global_state import _intra_bw_residual_frac

        v = encode_global_state(substrate, GlobalStateStats())
        per_domain_block = 7  # 7 fields per domain
        for i, s in enumerate(mdo_summaries):
            offset = i * per_domain_block
            expected_cpu_frac = (
                s.cpu_residual / s.cpu_capacity if s.cpu_capacity > 0 else 0.0
            )
            assert v[offset].item() == pytest.approx(expected_cpu_frac)
