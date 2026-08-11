"""Tests for intra-domain and cross-domain flow routing."""

from __future__ import annotations

import math

from orion.actors.routing import (
    RoutingSelector,
    allocate_route_bw,
    deallocate_route_bw,
    route_cross_domain_flow,
    route_flow,
)
from orion.types import LinkType


class TestRouteFlow:

    def test_same_node_no_routing(self, small_substrate):
        """Co-located VNFs need no routing."""
        node_ids = small_substrate.nodes_in_domain(0)
        result = route_flow(
            small_substrate, node_ids[0], node_ids[0],
            bw_demand=100.0, delay_budget=50.0,
            domain_node_ids=node_ids,
        )
        assert result.feasible
        assert result.path_links == []
        assert result.propagation_delay == 0.0

    def test_feasible_route_between_connected_nodes(self, small_substrate):
        """Should find a route between connected nodes in the same domain."""
        node_ids = small_substrate.nodes_in_domain(0)
        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=10.0, delay_budget=100.0,
            domain_node_ids=node_ids,
        )
        assert result.feasible
        assert len(result.path_links) > 0
        assert result.propagation_delay > 0.0

    def test_infeasible_bw_demand(self, small_substrate):
        """Route with excessive BW demand should fail."""
        node_ids = small_substrate.nodes_in_domain(0)
        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=999999.0, delay_budget=100.0,
            domain_node_ids=node_ids,
        )
        assert not result.feasible

    def test_infeasible_delay_budget(self, small_substrate):
        """Route with impossibly tight delay budget should fail."""
        node_ids = small_substrate.nodes_in_domain(0)
        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=10.0, delay_budget=0.001,
            domain_node_ids=node_ids,
        )
        assert not result.feasible

    def test_min_cost_selector_uses_log(self, small_substrate):
        """min_cost selector should use -log(residual/capacity) form."""
        node_ids = small_substrate.nodes_in_domain(0)
        selector = RoutingSelector("min_cost")

        # Get a known path
        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=10.0, delay_budget=100.0,
            domain_node_ids=node_ids,
            selector=selector,
        )
        if result.feasible and len(result.path_nodes) > 1:
            # Verify the cost is sum of -log(res/cap)
            expected = 0.0
            for i in range(len(result.path_nodes) - 1):
                u, v = result.path_nodes[i], result.path_nodes[i + 1]
                ed = small_substrate.graph.edges[u, v]
                ratio = ed["bw_residual"] / ed["bandwidth_capacity"]
                expected += -math.log(max(ratio, 1e-6))
            assert abs(result.cost - expected) < 1e-6

    def test_min_hops_selector(self, small_substrate):
        """min_hops selector should prefer shortest path by hop count."""
        node_ids = small_substrate.nodes_in_domain(0)
        selector = RoutingSelector("min_hops")

        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=10.0, delay_budget=100.0,
            domain_node_ids=node_ids,
            selector=selector,
        )
        if result.feasible:
            assert result.cost == len(result.path_links)


class TestBwAllocation:

    def test_allocate_decreases_residual(self, small_substrate):
        """allocate_route_bw should decrease link residual BW."""
        node_ids = small_substrate.nodes_in_domain(0)
        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=10.0, delay_budget=100.0,
            domain_node_ids=node_ids,
        )
        if not result.feasible:
            return

        # Record original BW
        original_bw = {}
        for link_id in result.path_links:
            original_bw[link_id] = small_substrate.get_residual_bw(link_id)

        allocate_route_bw(small_substrate, result.path_links, 10.0)

        for link_id in result.path_links:
            new_bw = small_substrate.get_residual_bw(link_id)
            assert abs(new_bw - (original_bw[link_id] - 10.0)) < 1e-6

    def test_deallocate_restores_residual(self, small_substrate):
        """deallocate_route_bw should restore link residual BW."""
        node_ids = small_substrate.nodes_in_domain(0)
        result = route_flow(
            small_substrate, node_ids[0], node_ids[-1],
            bw_demand=10.0, delay_budget=100.0,
            domain_node_ids=node_ids,
        )
        if not result.feasible:
            return

        original_bw = {}
        for link_id in result.path_links:
            original_bw[link_id] = small_substrate.get_residual_bw(link_id)

        allocate_route_bw(small_substrate, result.path_links, 10.0)
        deallocate_route_bw(small_substrate, result.path_links, 10.0)

        for link_id in result.path_links:
            restored_bw = small_substrate.get_residual_bw(link_id)
            assert abs(restored_bw - original_bw[link_id]) < 1e-6


class TestRouteCrossDomainFlow:
    """Cross-domain routing on the full substrate graph.

    small_substrate is a 3-domain chain: 0↔1↔2 (adjacent inter-domain links
    only). A flow from domain 0 to domain 2 must transit through domain 1.
    """

    def test_adjacent_domains_direct_route(self, small_substrate):
        """A (0,1) flow routes via inter-domain edges directly."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        d1_nodes = small_substrate.nodes_in_domain(1)
        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d1_nodes[0],
            bw_demand=10.0, delay_budget=500.0,
        )
        assert result.feasible
        assert len(result.path_links) > 0

        g = small_substrate.graph
        has_inter = False
        for link_id in result.path_links:
            for u, v, d in g.edges(data=True):
                if d["link_id"] == link_id:
                    lt = d["link_type"]
                    if lt == LinkType.INTER.value or lt == LinkType.INTER:
                        has_inter = True
                    break
        assert has_inter, "Path must include inter-domain edge"

    def test_non_adjacent_domains_transit(self, small_substrate):
        """A (0,2) flow on the chain must transit through domain 1."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        d2_nodes = small_substrate.nodes_in_domain(2)
        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d2_nodes[0],
            bw_demand=10.0, delay_budget=500.0,
        )
        assert result.feasible, "Must find multi-hop path through domain 1"
        assert len(result.path_links) >= 3, "At least 2 inter-domain + 1 transit link"

        g = small_substrate.graph
        inter_edges = []
        domains_traversed = set()
        for link_id in result.path_links:
            for u, v, d in g.edges(data=True):
                if d["link_id"] == link_id:
                    domains_traversed.add(g.nodes[u]["domain_id"])
                    domains_traversed.add(g.nodes[v]["domain_id"])
                    lt = d["link_type"]
                    if lt == LinkType.INTER.value or lt == LinkType.INTER:
                        inter_edges.append((u, v))
                    break

        assert 1 in domains_traversed, "Path must transit through domain 1"
        assert len(inter_edges) >= 2, "Need at least 0→1 and 1→2 inter-domain edges"

    def test_non_adjacent_bw_debit(self, small_substrate):
        """After allocating a (0,2) route, inter-domain edge residuals decrease."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        d2_nodes = small_substrate.nodes_in_domain(2)
        bw = 50.0

        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d2_nodes[0],
            bw_demand=bw, delay_budget=500.0,
        )
        assert result.feasible

        original_bw = {}
        for link_id in result.path_links:
            original_bw[link_id] = small_substrate.get_residual_bw(link_id)

        allocate_route_bw(small_substrate, result.path_links, bw)

        for link_id in result.path_links:
            new_bw = small_substrate.get_residual_bw(link_id)
            assert abs(new_bw - (original_bw[link_id] - bw)) < 1e-6, (
                f"Edge {link_id}: expected {original_bw[link_id] - bw}, got {new_bw}"
            )

    def test_non_adjacent_bw_debit_and_restore(self, small_substrate):
        """Allocate + deallocate on a (0,2) route restores all residuals."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        d2_nodes = small_substrate.nodes_in_domain(2)
        bw = 50.0

        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d2_nodes[0],
            bw_demand=bw, delay_budget=500.0,
        )
        assert result.feasible

        original_bw = {}
        for link_id in result.path_links:
            original_bw[link_id] = small_substrate.get_residual_bw(link_id)

        allocate_route_bw(small_substrate, result.path_links, bw)
        deallocate_route_bw(small_substrate, result.path_links, bw)

        for link_id in result.path_links:
            assert abs(
                small_substrate.get_residual_bw(link_id) - original_bw[link_id]
            ) < 1e-6

    def test_transit_capacity_exhausted(self, small_substrate):
        """A (0,2) flow exceeding transit BW must fail."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        d2_nodes = small_substrate.nodes_in_domain(2)
        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d2_nodes[0],
            bw_demand=999999.0, delay_budget=500.0,
        )
        assert not result.feasible

    def test_delay_budget_enforced(self, small_substrate):
        """Cross-domain route with tight delay budget must fail."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        d2_nodes = small_substrate.nodes_in_domain(2)
        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d2_nodes[0],
            bw_demand=10.0, delay_budget=0.001,
        )
        assert not result.feasible

    def test_same_node_no_routing(self, small_substrate):
        """Same-node cross-domain call should return empty path."""
        d0_nodes = small_substrate.nodes_in_domain(0)
        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d0_nodes[0],
            bw_demand=10.0, delay_budget=500.0,
        )
        assert result.feasible
        assert result.path_links == []

    def test_path_in_bw_allocations_after_commit(self, small_substrate):
        """The acceptance test: after routing and allocating a (0,2) flow,
        specific physical edges are debited and residuals decrease.

        This is the edge-residual-motion analogue of parameter-motion:
        prove the inter-domain edges actually get charged.
        """
        d0_nodes = small_substrate.nodes_in_domain(0)
        d2_nodes = small_substrate.nodes_in_domain(2)
        bw = 100.0

        # Snapshot ALL edge residuals
        g = small_substrate.graph
        pre_residuals = {
            d["link_id"]: d["bw_residual"]
            for _, _, d in g.edges(data=True)
        }

        result = route_cross_domain_flow(
            small_substrate, d0_nodes[0], d2_nodes[0],
            bw_demand=bw, delay_budget=500.0,
        )
        assert result.feasible

        allocate_route_bw(small_substrate, result.path_links, bw)

        # Verify: exactly the path edges decreased, others unchanged
        charged_links = set(result.path_links)
        for _, _, d in g.edges(data=True):
            lid = d["link_id"]
            if lid in charged_links:
                assert abs(d["bw_residual"] - (pre_residuals[lid] - bw)) < 1e-6, (
                    f"Edge {lid} should have been charged {bw}"
                )
            else:
                assert abs(d["bw_residual"] - pre_residuals[lid]) < 1e-6, (
                    f"Edge {lid} should NOT have been charged"
                )
