"""Shared fixtures for sim tests."""

from __future__ import annotations

import numpy as np
import pytest

from orion.config import TopologyConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.sim.slice_generator import generate_slice_request
from orion.types import SliceRequest


@pytest.fixture
def small_substrate() -> SubstrateNetwork:
    """3-domain, 5 nodes/domain substrate."""
    rng = np.random.default_rng(42)
    config = TopologyConfig(
        num_domains=3,
        nodes_per_domain=[5, 5, 5],
        intra_link_density=0.6,
        inter_domain_links=2,
    )
    return generate_multi_domain_topology(config, rng)


@pytest.fixture
def sample_slice(small_substrate) -> SliceRequest:
    """A deterministic slice request usable for greedy/verifier tests."""
    rng = np.random.default_rng(7)
    return generate_slice_request(
        request_id="req_test_0001",
        substrate=small_substrate,
        rng=rng,
    )
