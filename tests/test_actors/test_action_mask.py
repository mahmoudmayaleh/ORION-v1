"""Tests for action masking logic."""

from __future__ import annotations

import torch

from orion.actors.action_mask import compute_action_mask
from orion.actors.types import VNFAssignment
from orion.types import InfrastructureTier


class TestComputeActionMask:

    def test_valid_nodes_are_unmasked(self, small_substrate):
        """Nodes with correct tier and sufficient resources should be True."""
        domain_id = 0
        node_ids = sorted(small_substrate.nodes_in_domain(domain_id))
        g = small_substrate.graph

        # Find a tier that exists in domain 0
        tier = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
        permitted = [n for n in node_ids if g.nodes[n]["tier"] == tier.value]

        vnf = VNFAssignment(
            vnf_id="f_test", vnf_type="Test",
            cpu_demand=1.0, ram_demand=1.0,
            required_tier=tier,
            computational_intensity=1.0, vcr=1.0,
            bandwidth_in=10.0,
            permitted_nodes=permitted,
            position_in_sfc=0, sfc_length=1,
        )

        mask = compute_action_mask(small_substrate, node_ids, vnf)
        assert mask.any(), "At least one node should be valid"
        assert mask.sum().item() == len(permitted)

    def test_insufficient_resources_masked(self, small_substrate):
        """Nodes without enough CPU/RAM should be masked out."""
        domain_id = 0
        node_ids = sorted(small_substrate.nodes_in_domain(domain_id))
        g = small_substrate.graph

        tier = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
        permitted = [n for n in node_ids if g.nodes[n]["tier"] == tier.value]

        vnf = VNFAssignment(
            vnf_id="f_huge", vnf_type="Huge",
            cpu_demand=9999.0, ram_demand=9999.0,
            required_tier=tier,
            computational_intensity=1.0, vcr=1.0,
            bandwidth_in=10.0,
            permitted_nodes=permitted,
            position_in_sfc=0, sfc_length=1,
        )

        mask = compute_action_mask(small_substrate, node_ids, vnf)
        assert not mask.any(), "No node should have 9999 CPU"

    def test_wrong_tier_masked(self, small_substrate):
        """Nodes of wrong tier should always be masked."""
        domain_id = 0
        node_ids = sorted(small_substrate.nodes_in_domain(domain_id))
        g = small_substrate.graph

        # Use a tier that doesn't exist or restrict permitted_nodes to empty
        vnf = VNFAssignment(
            vnf_id="f_test", vnf_type="Test",
            cpu_demand=1.0, ram_demand=1.0,
            required_tier=InfrastructureTier.CENTRAL_CLOUD,
            computational_intensity=1.0, vcr=1.0,
            bandwidth_in=10.0,
            permitted_nodes=[],  # empty permitted set
            position_in_sfc=0, sfc_length=1,
        )

        mask = compute_action_mask(small_substrate, node_ids, vnf)
        assert not mask.any()

    def test_resource_overrides(self, small_substrate):
        """Autoregressive overrides should be respected in masking."""
        domain_id = 0
        node_ids = sorted(small_substrate.nodes_in_domain(domain_id))
        g = small_substrate.graph

        tier = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
        permitted = [n for n in node_ids if g.nodes[n]["tier"] == tier.value]

        vnf = VNFAssignment(
            vnf_id="f_test", vnf_type="Test",
            cpu_demand=2.0, ram_demand=2.0,
            required_tier=tier,
            computational_intensity=1.0, vcr=1.0,
            bandwidth_in=10.0,
            permitted_nodes=permitted,
            position_in_sfc=0, sfc_length=1,
        )

        # Override first permitted node to have insufficient resources
        overrides = {permitted[0]: (0.5, 0.5)}
        mask = compute_action_mask(
            small_substrate, node_ids, vnf,
            resource_overrides=overrides,
        )

        idx = node_ids.index(permitted[0])
        assert not mask[idx], "Overridden node should be masked"

    def test_mask_shape_matches_node_count(self, small_substrate):
        """Mask length should equal number of domain nodes."""
        domain_id = 0
        node_ids = sorted(small_substrate.nodes_in_domain(domain_id))
        g = small_substrate.graph

        tier = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
        permitted = [n for n in node_ids if g.nodes[n]["tier"] == tier.value]

        vnf = VNFAssignment(
            vnf_id="f_test", vnf_type="Test",
            cpu_demand=1.0, ram_demand=1.0,
            required_tier=tier,
            computational_intensity=1.0, vcr=1.0,
            bandwidth_in=10.0,
            permitted_nodes=permitted,
            position_in_sfc=0, sfc_length=1,
        )

        mask = compute_action_mask(small_substrate, node_ids, vnf)
        assert mask.shape == (len(node_ids),)
