"""Shared fixtures for domain actor tests.

All fixtures use fixed seeds for deterministic reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from orion.actors.types import PlanFragment, VNFAssignment
from orion.config import TopologyConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import FlowEdge, InfrastructureTier


@pytest.fixture
def seed():
    """Fixed seed for all actor tests."""
    return 42


@pytest.fixture
def small_substrate(seed) -> SubstrateNetwork:
    """3-domain substrate with 5 nodes per domain (small, fast tests)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = TopologyConfig(
        num_domains=3,
        nodes_per_domain=[5, 5, 5],
        intra_link_density=0.5,
        inter_domain_links=2,
    )
    return generate_multi_domain_topology(config, rng)


@pytest.fixture
def single_domain_substrate(seed) -> SubstrateNetwork:
    """1-domain substrate with 8 nodes for isolated domain actor tests."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = TopologyConfig(
        num_domains=1,
        nodes_per_domain=[8],
        intra_link_density=0.6,
        inter_domain_links=0,
    )
    return generate_multi_domain_topology(config, rng)


def _find_nodes_by_tier(substrate: SubstrateNetwork, domain_id: int, tier: str) -> list[str]:
    """Find nodes in a domain matching a tier."""
    return [
        n for n in substrate.nodes_in_domain(domain_id)
        if substrate.graph.nodes[n]["tier"] == tier
    ]


@pytest.fixture
def simple_fragment(small_substrate) -> PlanFragment:
    """A 2-VNF fragment for domain 0 with one intra-domain flow.

    Dynamically selects tiers that actually exist in the generated domain.
    """
    domain_id = 0
    node_ids = small_substrate.nodes_in_domain(domain_id)
    g = small_substrate.graph

    # Pick the tier of the first two nodes in domain 0
    tier0 = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
    tier1 = InfrastructureTier(g.nodes[node_ids[1]]["tier"])

    # Get permitted nodes for each tier
    permitted0 = [n for n in node_ids if g.nodes[n]["tier"] == tier0.value]
    permitted1 = [n for n in node_ids if g.nodes[n]["tier"] == tier1.value]

    vnf0 = VNFAssignment(
        vnf_id="f0", vnf_type="Firewall",
        cpu_demand=2.0, ram_demand=4.0,
        required_tier=tier0,
        computational_intensity=0.8, vcr=1.0,
        bandwidth_in=100.0,
        permitted_nodes=permitted0,
        position_in_sfc=0, sfc_length=2,
    )
    vnf1 = VNFAssignment(
        vnf_id="f1", vnf_type="vEPC",
        cpu_demand=4.0, ram_demand=8.0,
        required_tier=tier1,
        computational_intensity=1.0, vcr=1.0,
        bandwidth_in=100.0,
        permitted_nodes=permitted1,
        position_in_sfc=1, sfc_length=2,
    )

    return PlanFragment(
        domain_id=domain_id,
        vnf_assignments=[vnf0, vnf1],
        intra_flows=[FlowEdge(source_vnf="f0", target_vnf="f1", bandwidth_demand=100.0)],
        delay_budget_ms=50.0,
    )


@pytest.fixture
def cross_domain_fragment(small_substrate) -> PlanFragment:
    """A 2-VNF fragment for domain 0 where VNF f1's successor is in domain 1.

    Tests border-node awareness features.
    """
    domain_id = 0
    node_ids = small_substrate.nodes_in_domain(domain_id)
    g = small_substrate.graph

    tier0 = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
    tier1 = InfrastructureTier(g.nodes[node_ids[1]]["tier"])

    permitted0 = [n for n in node_ids if g.nodes[n]["tier"] == tier0.value]
    permitted1 = [n for n in node_ids if g.nodes[n]["tier"] == tier1.value]

    vnf0 = VNFAssignment(
        vnf_id="f0", vnf_type="Firewall",
        cpu_demand=2.0, ram_demand=4.0,
        required_tier=tier0,
        computational_intensity=0.8, vcr=1.0,
        bandwidth_in=100.0,
        permitted_nodes=permitted0,
        position_in_sfc=0, sfc_length=3,
        adjacent_domain_ids=set(),
    )
    vnf1 = VNFAssignment(
        vnf_id="f1", vnf_type="vEPC",
        cpu_demand=3.0, ram_demand=6.0,
        required_tier=tier1,
        computational_intensity=1.0, vcr=1.0,
        bandwidth_in=100.0,
        permitted_nodes=permitted1,
        position_in_sfc=1, sfc_length=3,
        adjacent_domain_ids={1},
    )

    return PlanFragment(
        domain_id=domain_id,
        vnf_assignments=[vnf0, vnf1],
        intra_flows=[FlowEdge(source_vnf="f0", target_vnf="f1", bandwidth_demand=100.0)],
        delay_budget_ms=50.0,
        target_domain_ids={1},
    )


@pytest.fixture
def empty_fragment() -> PlanFragment:
    """Empty fragment — MDO assigned zero VNFs to this domain."""
    return PlanFragment(
        domain_id=2,
        vnf_assignments=[],
        intra_flows=[],
        delay_budget_ms=50.0,
    )


@pytest.fixture
def infeasible_fragment(small_substrate) -> PlanFragment:
    """Fragment requiring more resources than any node has."""
    domain_id = 0
    node_ids = small_substrate.nodes_in_domain(domain_id)
    g = small_substrate.graph

    tier0 = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
    permitted = [n for n in node_ids if g.nodes[n]["tier"] == tier0.value]

    vnf = VNFAssignment(
        vnf_id="f_huge", vnf_type="MassiveVNF",
        cpu_demand=9999.0, ram_demand=9999.0,
        required_tier=tier0,
        computational_intensity=1.0, vcr=1.0,
        bandwidth_in=100.0,
        permitted_nodes=permitted,
        position_in_sfc=0, sfc_length=1,
    )

    return PlanFragment(
        domain_id=domain_id,
        vnf_assignments=[vnf],
        intra_flows=[],
        delay_budget_ms=50.0,
    )
