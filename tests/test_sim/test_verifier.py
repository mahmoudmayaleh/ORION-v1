"""Post-commit verifier tests.

Validates:
    - Clean placements pass.
    - C2/C3 fire when residuals go negative.
    - C5 fires on link over-allocation.
    - C7 fires under load-dependent sojourn even when light-load delay
      would have been within budget (the gap that motivates C1 + E1).
    - C9 fires when too many inter-domain hops were used.
    - hard_penalty_fired surfaces only C2/C3/C5b/C7 per v6.2 Eq. 9.
"""

from __future__ import annotations

import pytest

from orion.baselines.greedy_ffd import greedy_place_on_substrate
from orion.sim.verifier import GroundTruthVerdict, verify_committed_plan


@pytest.fixture
def admitted_state(small_substrate, sample_slice):
    """Place the sample slice via greedy and return (substrate, plan, slice).

    With the slice generator's β_in coupling fix in place, `min_throughput`
    matches the slice's flow demands by construction; no override needed.
    The delay budget is widened so the (already small) static delay sits
    well within budget — the test for clean placement is about the verdict
    not violating anything, not about a tight delay margin.
    """
    sample_slice.qos.max_e2e_delay = max(sample_slice.qos.max_e2e_delay, 1000.0)
    result = greedy_place_on_substrate(small_substrate, sample_slice)
    if not result.feasible:
        pytest.skip("greedy could not place sample slice on this substrate")
    return small_substrate, result.plan, sample_slice


class TestCleanPlacement:
    def test_no_violations_on_lightly_loaded_substrate(self, admitted_state) -> None:
        substrate, plan, slice_req = admitted_state
        verdict = verify_committed_plan(substrate, plan, slice_req,
                                        max_inter_domain_hops=10)
        assert verdict.feasible
        assert verdict.violated == []
        assert verdict.hard_penalty_fired is False
        # Sanity: delay computed and within budget.
        assert verdict.details["e2e_delay"] < verdict.details["delay_budget"]


class TestCapacityViolations:
    def test_c2_fires_on_cpu_overflow(self, admitted_state) -> None:
        substrate, plan, slice_req = admitted_state
        # Force a residual negative on one of the placed nodes.
        placed_node = next(iter(plan.vnf_placements.values()))
        substrate.graph.nodes[placed_node]["cpu_residual"] = -1.0
        verdict = verify_committed_plan(substrate, plan, slice_req)
        assert "C2" in verdict.violated
        assert verdict.hard_penalty_fired is True

    def test_c3_fires_on_ram_overflow(self, admitted_state) -> None:
        substrate, plan, slice_req = admitted_state
        placed_node = next(iter(plan.vnf_placements.values()))
        substrate.graph.nodes[placed_node]["ram_residual"] = -1.0
        verdict = verify_committed_plan(substrate, plan, slice_req)
        assert "C3" in verdict.violated
        assert verdict.hard_penalty_fired is True

    def test_c5_fires_on_link_overflow(self, admitted_state) -> None:
        substrate, plan, slice_req = admitted_state
        if not any(plan.flow_routes.values()):
            pytest.skip("slice was same-node placement; no link to overflow")
        # Pick the first link of the first non-empty route.
        link_id = next(
            lid for route in plan.flow_routes.values() if route for lid in route
        )
        for _, _, d in substrate.graph.edges(data=True):
            if d["link_id"] == link_id:
                d["bw_residual"] = -1.0
                break
        verdict = verify_committed_plan(substrate, plan, slice_req)
        assert "C5" in verdict.violated
        # C5 alone should NOT trigger the hard penalty per v6.2 Eq. 9.
        c_only = [c for c in verdict.violated if c != "C5"]
        if not c_only:
            assert verdict.hard_penalty_fired is False


class TestDelayUnderLoad:
    def test_c7_fires_when_node_is_saturated(self, admitted_state) -> None:
        """Saturate a placed node's CPU → M/M/1 sojourn → ∞ → C7."""
        substrate, plan, slice_req = admitted_state
        placed_node = next(iter(plan.vnf_placements.values()))
        # Drive residual to zero so cpu_used == capacity (saturation).
        substrate.graph.nodes[placed_node]["cpu_residual"] = 0.0
        verdict = verify_committed_plan(substrate, plan, slice_req)
        assert "C7" in verdict.violated
        assert verdict.hard_penalty_fired is True


class TestHopLimit:
    def test_c9_fires_when_hop_limit_zero(self, admitted_state) -> None:
        substrate, plan, slice_req = admitted_state
        verdict = verify_committed_plan(
            substrate, plan, slice_req, max_inter_domain_hops=0
        )
        # If any inter-domain hop exists, C9 must fire; if not, this is a
        # same-domain placement and the test is vacuous.
        if verdict.details["inter_domain_hops"] > 0:
            assert "C9" in verdict.violated
            assert verdict.hard_penalty_fired is False  # C9 is not in the penalty set


class TestVerdictShape:
    def test_details_always_populated(self, admitted_state) -> None:
        substrate, plan, slice_req = admitted_state
        verdict = verify_committed_plan(substrate, plan, slice_req)
        for k in ("e2e_delay", "delay_budget", "delay_slack",
                  "achieved_throughput", "throughput_floor",
                  "inter_domain_hops", "hop_limit"):
            assert k in verdict.details

    def test_hard_penalty_only_for_specified_set(self) -> None:
        # synthetic verdicts to lock in the policy without touching the substrate
        assert GroundTruthVerdict(False, ["C2"]).hard_penalty_fired
        assert GroundTruthVerdict(False, ["C3"]).hard_penalty_fired
        assert GroundTruthVerdict(False, ["C5b"]).hard_penalty_fired
        assert GroundTruthVerdict(False, ["C7"]).hard_penalty_fired
        assert not GroundTruthVerdict(False, ["C5"]).hard_penalty_fired
        assert not GroundTruthVerdict(False, ["C9"]).hard_penalty_fired
