"""Unit tests for FeasibilityChecker.

  - check():            C2, C3, C5, C5b, C7
  - check_structural(): C1, C4, C6, C8
"""

from __future__ import annotations

import networkx as nx

from orion.milp.feasibility import FeasibilityChecker
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import (
    VNF,
    FlowEdge,
    PlacementPlan,
    QoSRequirements,
    SliceRequest,
    SliceType,
)


def _make_substrate() -> SubstrateNetwork:
    G = nx.DiGraph()
    G.add_node("n0", domain_id=0, tier="mec", cpu_capacity=16.0, ram_capacity=64.0,
               processing_delay=1.0, cpu_residual=16.0, ram_residual=64.0)
    G.add_node("n1", domain_id=0, tier="regional_cloud", cpu_capacity=16.0, ram_capacity=64.0,
               processing_delay=1.5, cpu_residual=16.0, ram_residual=64.0)
    G.add_edge("n0", "n1", link_id="l_n0_n1", bandwidth_capacity=1000.0, bw_residual=1000.0,
               propagation_delay=2.0, link_type="intra")
    G.add_edge("n1", "n0", link_id="l_n1_n0", bandwidth_capacity=1000.0, bw_residual=1000.0,
               propagation_delay=2.0, link_type="intra")
    return SubstrateNetwork(graph=G, num_domains=1)


def _make_valid_plan() -> tuple[PlacementPlan, SliceRequest]:
    """A plan satisfying all constraints on the test substrate."""
    plan = PlacementPlan(
        plan_id="p_valid",
        vnf_placements={"f0": "n0", "f1": "n1"},
        cpu_allocations={"f0": 4.0, "f1": 4.0},
        ram_allocations={"f0": 8.0, "f1": 8.0},
        flow_routes={("f0", "f1"): ["l_n0_n1"]},
        bw_allocations={("f0", "f1"): {"l_n0_n1": 100.0}},
    )
    request = SliceRequest(
        request_id="req_valid",
        slice_type=SliceType.EMBB,
        vnfs=[
            VNF("f0", "Firewall", 4.0, 8.0, ["n0", "n1"]),
            VNF("f1", "vEPC", 4.0, 8.0, ["n0", "n1"]),
        ],
        flow_edges=[FlowEdge("f0", "f1", 100.0)],
        qos=QoSRequirements(50.0, 50.0),
        arrival_time=0.0,
        lifetime=0.0,
    )
    return plan, request


# ── Module 3 LP-check (C2, C3, C5, C7) ────────────────────────────────────────

class TestValidPlan:
    def test_valid_plan_passes(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        result = checker.check(plan, req)
        assert result.is_feasible
        assert result.violated_constraints == []


class TestC2CpuViolation:
    def test_cpu_overallocation_caught(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        plan.cpu_allocations["f0"] = 20.0  # n0 has 16 CPU residual
        result = checker.check(plan, req)
        assert not result.is_feasible
        assert any("C2" in v for v in result.violated_constraints)

    def test_ram_overallocation_caught(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        plan.ram_allocations["f1"] = 100.0  # n1 has 64 GB residual
        result = checker.check(plan, req)
        assert not result.is_feasible
        assert any("C3" in v for v in result.violated_constraints)


class TestC5BwViolation:
    def test_bw_overallocation_caught(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        plan.bw_allocations[("f0", "f1")]["l_n0_n1"] = 1500.0  # > 1000 Mbps
        result = checker.check(plan, req)
        assert not result.is_feasible
        assert any("C5" in v for v in result.violated_constraints)


class TestC5bThroughputFloor:
    def test_low_throughput_caught(self) -> None:
        """Flow with total BW below min_throughput should fail C5b."""
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        # min_throughput is 50.0 Mbps, set allocated BW to 10.0
        plan.bw_allocations[("f0", "f1")]["l_n0_n1"] = 10.0
        result = checker.check(plan, req)
        assert not result.is_feasible
        assert any("C5b" in v for v in result.violated_constraints)

    def test_sufficient_throughput_passes(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        # Default plan has 100 Mbps >= 50 min_throughput, should pass
        result = checker.check(plan, req)
        assert result.is_feasible


class TestC7DelayViolation:
    def test_delay_budget_exceeded(self) -> None:
        # n0 proc=1ms, n1 proc=1ms; link=30ms; budget=5ms -> C7 violated
        G = nx.DiGraph()
        G.add_node("n0", domain_id=0, tier="mec", cpu_capacity=16.0, ram_capacity=64.0,
                   processing_delay=1.0, cpu_residual=16.0, ram_residual=64.0)
        G.add_node("n1", domain_id=0, tier="mec", cpu_capacity=16.0, ram_capacity=64.0,
                   processing_delay=1.0, cpu_residual=16.0, ram_residual=64.0)
        G.add_edge("n0", "n1", link_id="l_slow", bandwidth_capacity=1000.0,
                   bw_residual=1000.0, propagation_delay=30.0, link_type="intra")
        G.add_edge("n1", "n0", link_id="l_slow_r", bandwidth_capacity=1000.0,
                   bw_residual=1000.0, propagation_delay=30.0, link_type="intra")

        checker = FeasibilityChecker(SubstrateNetwork(graph=G, num_domains=1))
        plan = PlacementPlan(
            plan_id="p_delay",
            vnf_placements={"f0": "n0", "f1": "n1"},
            cpu_allocations={"f0": 2.0, "f1": 2.0},
            ram_allocations={"f0": 4.0, "f1": 4.0},
            flow_routes={("f0", "f1"): ["l_slow"]},
            bw_allocations={("f0", "f1"): {"l_slow": 50.0}},
        )
        req = SliceRequest(
            request_id="req_delay",
            slice_type=SliceType.URLLC,
            vnfs=[VNF("f0", "F", 2.0, 4.0, ["n0", "n1"]), VNF("f1", "V", 2.0, 4.0, ["n0", "n1"])],
            flow_edges=[FlowEdge("f0", "f1", 50.0)],
            qos=QoSRequirements(max_e2e_delay=5.0, min_throughput=10.0),
            arrival_time=0.0,
            lifetime=0.0,
        )
        result = checker.check(plan, req)
        assert not result.is_feasible
        assert "C7_delay" in result.violated_constraints
        assert result.violation_details["C7_delay"] > 0.0


# ── Agent B structural checker (C1, C4, C6, C8) ───────────────────────────────

class TestStructuralC1PlacementUniqueness:
    def test_missing_vnf_placement_caught(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        del plan.vnf_placements["f1"]
        result = checker.check_structural(plan, req)
        assert not result.is_feasible
        assert any("C1" in v for v in result.violated_constraints)


class TestStructuralC4ResourceSufficiency:
    def test_cpu_below_demand_caught(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        plan.cpu_allocations["f0"] = 1.0  # f0 needs 4.0
        result = checker.check_structural(plan, req)
        assert not result.is_feasible
        assert any("C4_cpu_f0" in v for v in result.violated_constraints)

    def test_ram_below_demand_caught(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        plan.ram_allocations["f1"] = 1.0  # f1 needs 8.0
        result = checker.check_structural(plan, req)
        assert not result.is_feasible
        assert any("C4_ram_f1" in v for v in result.violated_constraints)


class TestStructuralC6FlowConservation:
    def test_broken_route_caught(self) -> None:
        """A route using a non-existent link_id should fail C6."""
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        plan.flow_routes[("f0", "f1")] = ["l_nonexistent"]
        result = checker.check_structural(plan, req)
        assert not result.is_feasible
        assert any("C6" in v for v in result.violated_constraints)


class TestStructuralC8PlacementRules:
    def test_wrong_node_caught(self) -> None:
        """Placing a VNF on a node not in its permitted_nodes should fail C8."""
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        req.vnfs[0].permitted_nodes.clear()
        req.vnfs[0].permitted_nodes.append("n1")
        plan.vnf_placements["f0"] = "n0"  # n0 not permitted
        result = checker.check_structural(plan, req)
        assert not result.is_feasible
        assert any("C8" in v for v in result.violated_constraints)


class TestStructuralValidPlan:
    def test_valid_plan_passes_structural(self) -> None:
        checker = FeasibilityChecker(_make_substrate())
        plan, req = _make_valid_plan()
        result = checker.check_structural(plan, req)
        assert result.is_feasible
        assert result.violated_constraints == []
