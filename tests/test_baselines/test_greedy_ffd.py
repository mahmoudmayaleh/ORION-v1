"""Greedy FFD baseline tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orion.baselines.greedy_ffd import (
    GreedyConfig,
    compute_cost_greedy,
    greedy_place_on_substrate,
)
from orion.config import TopologyConfig
from orion.sim.slice_generator import generate_slice_request
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology


@pytest.fixture
def substrate() -> SubstrateNetwork:
    rng = np.random.default_rng(42)
    return generate_multi_domain_topology(
        TopologyConfig(
            num_domains=3,
            nodes_per_domain=[5, 5, 5],
            intra_link_density=0.6,
            inter_domain_links=2,
        ),
        rng,
    )


@pytest.fixture
def slice_req(substrate):
    rng = np.random.default_rng(7)
    return generate_slice_request("req_g0001", substrate, rng)


class TestComputeCostGreedy:
    def test_returns_finite_cost_on_feasible_slice(self, substrate, slice_req) -> None:
        cost = compute_cost_greedy(substrate, slice_req)
        assert math.isfinite(cost)
        assert cost > 0

    def test_does_not_mutate_substrate(self, substrate, slice_req) -> None:
        cpu_before = {
            n: substrate.graph.nodes[n]["cpu_residual"]
            for n in substrate.graph.nodes
        }
        ram_before = {
            n: substrate.graph.nodes[n]["ram_residual"]
            for n in substrate.graph.nodes
        }
        bw_before = {
            d["link_id"]: d["bw_residual"]
            for _, _, d in substrate.graph.edges(data=True)
        }

        compute_cost_greedy(substrate, slice_req)

        for n in substrate.graph.nodes:
            assert substrate.graph.nodes[n]["cpu_residual"] == cpu_before[n]
            assert substrate.graph.nodes[n]["ram_residual"] == ram_before[n]
        for _, _, d in substrate.graph.edges(data=True):
            assert d["bw_residual"] == bw_before[d["link_id"]]

    def test_deterministic(self, substrate, slice_req) -> None:
        c1 = compute_cost_greedy(substrate, slice_req)
        c2 = compute_cost_greedy(substrate, slice_req)
        assert c1 == c2

    def test_infeasible_slice_returns_inf(self, substrate, slice_req) -> None:
        # Inflate first VNF demand beyond any node capacity
        slice_req.vnfs[0].cpu_demand = 1e9
        cost = compute_cost_greedy(substrate, slice_req)
        assert math.isinf(cost)


class TestGreedyPlaceOnSubstrate:
    def test_feasible_placement_mutates_residuals(self, substrate, slice_req) -> None:
        total_cpu_before = sum(
            substrate.graph.nodes[n]["cpu_residual"] for n in substrate.graph.nodes
        )
        result = greedy_place_on_substrate(substrate, slice_req)
        assert result.feasible
        assert result.plan is not None
        total_cpu_after = sum(
            substrate.graph.nodes[n]["cpu_residual"] for n in substrate.graph.nodes
        )
        total_cpu_demand = sum(v.cpu_demand for v in slice_req.vnfs)
        assert total_cpu_before - total_cpu_after == pytest.approx(total_cpu_demand)

    def test_infeasible_does_not_mutate(self, substrate, slice_req) -> None:
        slice_req.vnfs[0].cpu_demand = 1e9
        total_cpu_before = sum(
            substrate.graph.nodes[n]["cpu_residual"] for n in substrate.graph.nodes
        )
        result = greedy_place_on_substrate(substrate, slice_req)
        assert not result.feasible
        total_cpu_after = sum(
            substrate.graph.nodes[n]["cpu_residual"] for n in substrate.graph.nodes
        )
        assert total_cpu_before == pytest.approx(total_cpu_after)

    def test_cost_components_decompose(self, substrate, slice_req) -> None:
        cfg = GreedyConfig(alpha=2.0, gamma_intra=3.0, gamma_inter=5.0)
        result = greedy_place_on_substrate(substrate, slice_req, cfg)
        if not result.feasible:
            pytest.skip("slice infeasible under this seed")
        expected = (
            cfg.alpha * result.resource_cost
            + cfg.gamma_intra * result.intra_bw
            + cfg.gamma_inter * result.inter_bw
        )
        assert result.cost == pytest.approx(expected)
