"""Integration tests for the MILP solver on real (tiny) instances.

Per CLAUDE.md: do NOT mock the MILP solver. Use the real CBC solver on
tiny instances (2-node, 1-link substrate) so tests run in < 5 seconds.
"""

from __future__ import annotations

import networkx as nx
import pytest

from orion.config import MILPConfig
from orion.milp.solver import MILPSolver
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import (
    VNF,
    FlowEdge,
    QoSRequirements,
    SliceRequest,
    SliceType,
)

# ── Shared tiny fixtures ───────────────────────────────────────────────────────

def _make_two_node_substrate(
    cpu: float = 16.0,
    ram: float = 64.0,
    bw: float = 1000.0,
    delay: float = 2.0,
) -> SubstrateNetwork:
    """Build the minimal 2-node, 2-directed-link substrate for solver tests."""
    G = nx.DiGraph()
    G.add_node("n0", domain_id=0, tier="mec", cpu_capacity=cpu, ram_capacity=ram,
               processing_delay=1.0, cpu_residual=cpu, ram_residual=ram)
    G.add_node("n1", domain_id=0, tier="regional_cloud", cpu_capacity=cpu, ram_capacity=ram,
               processing_delay=1.5, cpu_residual=cpu, ram_residual=ram)
    G.add_edge("n0", "n1", link_id="l_n0_n1", bandwidth_capacity=bw, bw_residual=bw,
               propagation_delay=delay, link_type="intra")
    G.add_edge("n1", "n0", link_id="l_n1_n0", bandwidth_capacity=bw, bw_residual=bw,
               propagation_delay=delay, link_type="intra")
    return SubstrateNetwork(graph=G, num_domains=1)


def _make_simple_request(
    cpu_demand: float = 2.0,
    ram_demand: float = 4.0,
    bw_demand: float = 50.0,
    max_delay: float = 50.0,
    permitted: list[str] | None = None,
) -> SliceRequest:
    if permitted is None:
        permitted = ["n0", "n1"]
    return SliceRequest(
        request_id="req_test",
        slice_type=SliceType.EMBB,
        vnfs=[
            VNF("f0", "Firewall", cpu_demand, ram_demand, permitted),
            VNF("f1", "vEPC", cpu_demand, ram_demand, permitted),
        ],
        flow_edges=[FlowEdge("f0", "f1", bw_demand)],
        qos=QoSRequirements(max_delay, 10.0),
        arrival_time=0.0,
        lifetime=0.0,
    )


@pytest.fixture
def solver() -> MILPSolver:
    return MILPSolver(MILPConfig())


# ── Core correctness tests ─────────────────────────────────────────────────────

class TestTinyInstance:
    def test_tiny_instance_optimal(self, solver: MILPSolver) -> None:
        """Verify the solver returns Optimal on a feasible 2-node instance."""
        substrate = _make_two_node_substrate()
        slices = [_make_simple_request()]
        sol = solver.solve(substrate, slices)
        assert sol.status == "Optimal"
        assert sol.objective_value > 0.0

    def test_slice_admitted(self, solver: MILPSolver) -> None:
        substrate = _make_two_node_substrate()
        slices = [_make_simple_request()]
        sol = solver.solve(substrate, slices)
        assert sol.admitted["req_test"] is True
        assert "req_test" in sol.placements

    def test_c1_each_vnf_placed_once(self, solver: MILPSolver) -> None:
        """C1: Each VNF must appear exactly once in vnf_placements."""
        substrate = _make_two_node_substrate()
        slices = [_make_simple_request()]
        sol = solver.solve(substrate, slices)
        plan = sol.placements["req_test"]
        assert set(plan.vnf_placements.keys()) == {"f0", "f1"}
        # Each placement is a valid node
        assert all(n in ("n0", "n1") for n in plan.vnf_placements.values())

    def test_c4_resources_meet_demands(self, solver: MILPSolver) -> None:
        """C4: Allocated CPU and RAM must be >= demands for admitted VNFs."""
        substrate = _make_two_node_substrate()
        req = _make_simple_request(cpu_demand=3.0, ram_demand=6.0)
        sol = solver.solve(substrate, [req])
        plan = sol.placements["req_test"]
        for f in req.vnfs:
            assert plan.cpu_allocations[f.vnf_id] >= f.cpu_demand - 1e-4
            assert plan.ram_allocations[f.vnf_id] >= f.ram_demand - 1e-4

    def test_c6_flow_conservation(self, solver: MILPSolver) -> None:
        """C6: Route must connect the two placed VNF nodes."""
        substrate = _make_two_node_substrate()
        slices = [_make_simple_request()]
        sol = solver.solve(substrate, slices)
        plan = sol.placements["req_test"]
        f0_node = plan.vnf_placements["f0"]
        f1_node = plan.vnf_placements["f1"]
        route = plan.flow_routes.get(("f0", "f1"), [])
        if f0_node != f1_node:
            # Route must be non-empty and use the connecting link
            assert len(route) > 0

    def test_route_uses_substrate_links(self, solver: MILPSolver) -> None:
        """All links in any route must exist in the substrate graph."""
        substrate = _make_two_node_substrate()
        slices = [_make_simple_request()]
        sol = solver.solve(substrate, slices)
        plan = sol.placements["req_test"]
        known_links = {d["link_id"] for _, _, d in substrate.graph.edges(data=True)}
        for route_links in plan.flow_routes.values():
            for l_id in route_links:
                assert l_id in known_links, f"Unknown link in route: {l_id}"

    def test_objective_positive_on_feasible(self, solver: MILPSolver) -> None:
        substrate = _make_two_node_substrate()
        slices = [_make_simple_request()]
        sol = solver.solve(substrate, slices)
        # Revenue (mu=100) should dominate costs for a small request
        assert sol.objective_value > 0.0


class TestInfeasibleCases:
    def test_infeasible_rejected_cpu_overflow(self, solver: MILPSolver) -> None:
        """A request demanding more CPU than any node should be rejected (z=0)."""
        substrate = _make_two_node_substrate(cpu=4.0)
        req = _make_simple_request(cpu_demand=10.0)  # 10 > 4 on any node
        sol = solver.solve(substrate, [req])
        assert sol.admitted.get("req_test") is False

    def test_infeasible_no_permitted_nodes(self, solver: MILPSolver) -> None:
        """A VNF with no permitted nodes must result in rejection."""
        substrate = _make_two_node_substrate()
        req = _make_simple_request(permitted=[])
        # permitted=[] means D_f is empty; no x variables created -> C1 forces z=0
        # This may also result in an infeasible status depending on PuLP version
        sol = solver.solve(substrate, [req])
        assert not sol.admitted.get("req_test", True)

    def test_c7_tight_delay_respected(self, solver: MILPSolver) -> None:
        """With a 1ms delay budget, the slice cannot be routed (links add >= 2ms)."""
        substrate = _make_two_node_substrate(delay=5.0)
        # Node proc delays: n0=1.0ms, n1=1.5ms. Link delay=5ms.
        # Even co-located: 1.0+1.5 = 2.5ms > 1ms budget.
        req = _make_simple_request(max_delay=1.0)
        sol = solver.solve(substrate, [req])
        assert sol.admitted.get("req_test") is False


class TestMultiSlice:
    def test_two_slices_both_admitted_if_feasible(self, solver: MILPSolver) -> None:
        """Two lightweight slices should both be admitted on a 16-CPU substrate."""
        substrate = _make_two_node_substrate(cpu=16.0, ram=64.0)
        req1 = SliceRequest(
            request_id="req_A",
            slice_type=SliceType.URLLC,
            vnfs=[VNF("f0", "Firewall", 1.0, 2.0, ["n0", "n1"])],
            flow_edges=[],
            qos=QoSRequirements(50.0, 5.0),
            arrival_time=0.0,
            lifetime=0.0,
        )
        req2 = SliceRequest(
            request_id="req_B",
            slice_type=SliceType.EMBB,
            vnfs=[VNF("f0", "CDN", 1.0, 2.0, ["n0", "n1"])],
            flow_edges=[],
            qos=QoSRequirements(100.0, 10.0),
            arrival_time=0.0,
            lifetime=0.0,
        )
        sol = solver.solve(substrate, [req1, req2])
        assert sol.admitted["req_A"] is True
        assert sol.admitted["req_B"] is True

    def test_solve_time_recorded(self, solver: MILPSolver) -> None:
        substrate = _make_two_node_substrate()
        sol = solver.solve(substrate, [_make_simple_request()])
        assert sol.solve_time > 0.0
