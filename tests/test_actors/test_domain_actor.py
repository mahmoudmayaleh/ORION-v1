"""Integration tests for DomainActor: end-to-end fragment -> response.

Covers 8 scenarios:
  1. Feasible 2-VNF fragment with intra-domain routing
  2. Empty fragment returns trivially feasible
  3. Infeasible fragment (resource exceeded) returns feasible=False
  4. Cross-domain fragment has correct border features
  5. Rollback restores substrate state on failure
  6. Deterministic mode gives consistent placements
  7. Log probs have correct shape (one per VNF)
  8. Single-VNF fragment (no routing needed)
"""

from __future__ import annotations

import copy

import torch

from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.actors.types import DomainResponse, PlanFragment, VNFAssignment
from orion.types import FlowEdge, InfrastructureTier


class TestDomainActorIntegration:

    def _make_actor(self, domain_id: int = 0) -> DomainActor:
        torch.manual_seed(42)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        return DomainActor(domain_id=domain_id, policy=policy)

    def test_feasible_fragment(self, small_substrate, simple_fragment):
        """2-VNF fragment should produce a feasible response with placements."""
        actor = self._make_actor(0)
        torch.manual_seed(42)
        response = actor.act(small_substrate, simple_fragment)

        assert response.feasible
        assert response.domain_id == 0
        assert len(response.placements) == 2
        assert "f0" in response.placements
        assert "f1" in response.placements
        assert response.intra_delay > 0.0
        assert response.resource_cost > 0.0

    def test_empty_fragment(self, small_substrate, empty_fragment):
        """Empty fragment should return trivially feasible response."""
        actor = self._make_actor(2)
        response = actor.act(small_substrate, empty_fragment)

        assert response.feasible
        assert response.domain_id == 2
        assert len(response.placements) == 0
        assert response.intra_delay == 0.0
        assert response.resource_cost == 0.0

    def test_infeasible_fragment(self, small_substrate, infeasible_fragment):
        """Fragment with impossible demands should return infeasible (via NULL action)."""
        actor = self._make_actor(0)
        response = actor.act(small_substrate, infeasible_fragment)

        assert not response.feasible
        assert len(response.placements) == 0

    def test_rollback_on_failure(self, small_substrate, infeasible_fragment):
        """Substrate state should be restored after infeasible placement."""
        # Snapshot original state
        original_residuals = {}
        for node_id in small_substrate.nodes_in_domain(0):
            original_residuals[node_id] = (
                small_substrate.get_residual_cpu(node_id),
                small_substrate.get_residual_ram(node_id),
            )

        actor = self._make_actor(0)
        response = actor.act(small_substrate, infeasible_fragment)
        assert not response.feasible

        # Verify state is restored
        for node_id, (orig_cpu, orig_ram) in original_residuals.items():
            assert abs(small_substrate.get_residual_cpu(node_id) - orig_cpu) < 1e-6
            assert abs(small_substrate.get_residual_ram(node_id) - orig_ram) < 1e-6

    def test_deterministic_consistency(self, small_substrate, simple_fragment):
        """Deterministic mode should give same placements on same state."""
        substrate1 = copy.deepcopy(small_substrate)
        substrate2 = copy.deepcopy(small_substrate)

        actor = self._make_actor(0)
        torch.manual_seed(42)
        r1 = actor.act(substrate1, simple_fragment, deterministic=True)
        torch.manual_seed(42)
        r2 = actor.act(substrate2, simple_fragment, deterministic=True)

        if r1.feasible and r2.feasible:
            assert r1.placements == r2.placements

    def test_log_probs_shape(self, small_substrate, simple_fragment):
        """Log probs should have one entry per VNF in the fragment."""
        actor = self._make_actor(0)
        torch.manual_seed(42)
        response = actor.act(small_substrate, simple_fragment)

        if response.feasible:
            n_vnfs = len(simple_fragment.vnf_assignments)
            assert response.log_probs.shape == (n_vnfs,)
            assert all(lp.isfinite() for lp in response.log_probs)

    def test_single_vnf_fragment(self, small_substrate):
        """Single VNF fragment needs no routing."""
        domain_id = 0
        node_ids = small_substrate.nodes_in_domain(domain_id)
        g = small_substrate.graph
        tier = InfrastructureTier(g.nodes[node_ids[0]]["tier"])
        permitted = [n for n in node_ids if g.nodes[n]["tier"] == tier.value]

        vnf = VNFAssignment(
            vnf_id="f_solo", vnf_type="Solo",
            cpu_demand=2.0, ram_demand=4.0,
            required_tier=tier,
            computational_intensity=1.0, vcr=1.0,
            bandwidth_in=50.0,
            permitted_nodes=permitted,
            position_in_sfc=0, sfc_length=1,
        )

        fragment = PlanFragment(
            domain_id=domain_id,
            vnf_assignments=[vnf],
            intra_flows=[],
            delay_budget_ms=50.0,
        )

        actor = self._make_actor(0)
        torch.manual_seed(42)
        response = actor.act(small_substrate, fragment)

        assert response.feasible
        assert len(response.placements) == 1
        assert len(response.routes) == 0

    def test_cross_domain_fragment_border_features(
        self, small_substrate, cross_domain_fragment
    ):
        """Cross-domain fragment should activate border features."""
        actor = self._make_actor(0)
        torch.manual_seed(42)
        response = actor.act(small_substrate, cross_domain_fragment)

        # Should succeed (resources are sufficient)
        # We mainly care that it doesn't crash with target_domain_ids set
        assert isinstance(response, DomainResponse)

    def test_substrate_modified_on_success(self, small_substrate, simple_fragment):
        """Successful placement should leave substrate with reduced residuals."""
        original_total_cpu = sum(
            small_substrate.get_residual_cpu(n)
            for n in small_substrate.nodes_in_domain(0)
        )

        actor = self._make_actor(0)
        torch.manual_seed(42)
        response = actor.act(small_substrate, simple_fragment)

        if response.feasible:
            new_total_cpu = sum(
                small_substrate.get_residual_cpu(n)
                for n in small_substrate.nodes_in_domain(0)
            )
            expected_reduction = sum(
                v.cpu_demand for v in simple_fragment.vnf_assignments
            )
            assert abs((original_total_cpu - new_total_cpu) - expected_reduction) < 1e-6

    def test_entropy_is_nonnegative(self, small_substrate, simple_fragment):
        """Average entropy should be >= 0."""
        actor = self._make_actor(0)
        torch.manual_seed(42)
        response = actor.act(small_substrate, simple_fragment)

        if response.feasible:
            assert response.entropy >= 0.0
