"""Shared pytest fixtures for the ORION test suite.

All fixtures that require randomness use a fixed seed (42) for determinism.
Tests must not re-define fixtures already provided here.
"""

from __future__ import annotations

import numpy as np
import pytest

from orion.config import (
    OrionConfig,
)
from orion.types import (
    VNF,
    FlowEdge,
    InfrastructureTier,
    LinkAttributes,
    LinkType,
    NodeAttributes,
    QoSRequirements,
    SliceRequest,
    SliceType,
)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixed-seed random generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def orion_config() -> OrionConfig:
    """Minimal OrionConfig with deterministic seed."""
    return OrionConfig(seed=42)


@pytest.fixture
def simple_slice_request() -> SliceRequest:
    """A 2-VNF slice request compatible with two_node_substrate."""
    return SliceRequest(
        request_id="req_test_00001",
        slice_type=SliceType.EMBB,
        vnfs=[
            VNF(
                vnf_id="f0",
                vnf_type="Firewall",
                cpu_demand=2.0,
                ram_demand=4.0,
                permitted_nodes=["n0", "n1"],
            ),
            VNF(
                vnf_id="f1",
                vnf_type="vEPC",
                cpu_demand=4.0,
                ram_demand=8.0,
                permitted_nodes=["n0", "n1"],
            ),
        ],
        flow_edges=[
            FlowEdge(source_vnf="f0", target_vnf="f1", bandwidth_demand=100.0),
        ],
        qos=QoSRequirements(
            max_e2e_delay=50.0,
            min_throughput=50.0,
        ),
        arrival_time=0.0,
        lifetime=0.0,
    )
