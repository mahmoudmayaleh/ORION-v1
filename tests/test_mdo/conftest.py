"""Shared fixtures for MDO coordinator tests."""

import pytest
import torch

from orion.mdo.types import (
    DomainSummary,
    InterDomainLink,
    MDOObservation,
    PlanSummary,
    RetryHistory,
    ViolationInfo,
    PartitionAttempt,
    RewardComponents,
)
from orion.types import InfrastructureTier


@pytest.fixture
def three_domain_summaries():
    """Three domains: RAN/edge, MEC, Cloud."""
    return [
        DomainSummary(
            domain_id=0,
            dominant_tier=InfrastructureTier.RAN_EDGE,
            cpu_residual=50.0, ram_residual=100.0,
            cpu_capacity=100.0, ram_capacity=200.0,
            supported_tiers=[InfrastructureTier.RAN_EDGE, InfrastructureTier.MEC],
        ),
        DomainSummary(
            domain_id=1,
            dominant_tier=InfrastructureTier.MEC,
            cpu_residual=80.0, ram_residual=160.0,
            cpu_capacity=120.0, ram_capacity=240.0,
            supported_tiers=[InfrastructureTier.MEC, InfrastructureTier.REGIONAL_CLOUD],
        ),
        DomainSummary(
            domain_id=2,
            dominant_tier=InfrastructureTier.CENTRAL_CLOUD,
            cpu_residual=200.0, ram_residual=400.0,
            cpu_capacity=300.0, ram_capacity=600.0,
            supported_tiers=[InfrastructureTier.REGIONAL_CLOUD, InfrastructureTier.CENTRAL_CLOUD],
        ),
    ]


@pytest.fixture
def inter_domain_links():
    """Two inter-domain links: d0-d1, d1-d2."""
    return [
        InterDomainLink(source_domain=0, target_domain=1, bw_residual=500.0, bw_capacity=1000.0, propagation_delay=2.0),
        InterDomainLink(source_domain=1, target_domain=2, bw_residual=300.0, bw_capacity=800.0, propagation_delay=5.0),
    ]


@pytest.fixture
def simple_plan():
    """A 3-VNF linear SFC: RAN_EDGE -> MEC -> CENTRAL_CLOUD."""
    return PlanSummary(
        vnf_ids=["f0", "f1", "f2"],
        required_tiers=[
            InfrastructureTier.RAN_EDGE,
            InfrastructureTier.MEC,
            InfrastructureTier.CENTRAL_CLOUD,
        ],
        suggested_domains=[0, 1, 2],
        cpu_demands=[10.0, 20.0, 15.0],
        ram_demands=[20.0, 40.0, 30.0],
        vcrs=[1.0, 0.8, 1.2],
        bw_demands=[100.0, 80.0],
    )


@pytest.fixture
def simple_obs(three_domain_summaries, inter_domain_links, simple_plan):
    return MDOObservation(
        domain_summaries=three_domain_summaries,
        inter_domain_links=inter_domain_links,
        plan=simple_plan,
    )
