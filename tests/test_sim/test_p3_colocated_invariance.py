"""P3: co-located slices are byte-identical — cross-domain machinery is inert.

Strong-form state-equality test on a substrate that HAS inter-domain
structure (so the cross-domain machinery is present and could fire, but
doesn't for a co-located slice).

Assertions:
  1. inter_domain_demand_by_pair returns {} for a co-located partition.
  2. Every inter-domain residual is byte-identical before vs after allocate.
  3. The intra link IS correctly debited by beta.
  4. deallocate restores every residual exactly.
  5. Verifier reports inter_domain_hops == 0.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from orion.baselines.greedy_ffd import greedy_place_on_substrate
from orion.config import TopologyConfig
from orion.mdo.precommit_check import inter_domain_demand_by_pair
from orion.sim.verifier import verify_committed_plan
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import (
    VNF,
    FlowEdge,
    InfrastructureTier,
    QoSRequirements,
    SliceRequest,
    SliceType,
)


@pytest.fixture
def inter_domain_substrate() -> SubstrateNetwork:
    """2-domain substrate with inter-domain links — machinery is present."""
    rng = np.random.default_rng(99)
    config = TopologyConfig(
        num_domains=2,
        nodes_per_domain=[4, 4],
        intra_link_density=0.7,
        inter_domain_links=2,
    )
    return generate_multi_domain_topology(config, rng)


@pytest.fixture
def colocated_slice(inter_domain_substrate: SubstrateNetwork) -> SliceRequest:
    """2-VNF slice whose permitted_nodes are ALL in domain 0 only."""
    sub = inter_domain_substrate
    dom0_nodes = sub.nodes_in_domain(0)
    return SliceRequest(
        request_id="req_p3_coloc",
        slice_type=SliceType.EMBB,
        vnfs=[
            VNF(
                vnf_id="p3_f0",
                vnf_type="Firewall",
                cpu_demand=1.0,
                ram_demand=1.0,
                permitted_nodes=dom0_nodes,
                computational_intensity=0.5,
                vcr=1.0,
            ),
            VNF(
                vnf_id="p3_f1",
                vnf_type="vEPC",
                cpu_demand=1.0,
                ram_demand=1.0,
                permitted_nodes=dom0_nodes,
                computational_intensity=0.5,
                vcr=1.0,
            ),
        ],
        flow_edges=[
            FlowEdge(
                source_vnf="p3_f0",
                target_vnf="p3_f1",
                bandwidth_demand=10.0,
            ),
        ],
        qos=QoSRequirements(max_e2e_delay=100.0, min_throughput=10.0),
        arrival_time=0.0,
        lifetime=10.0,
    )


def _snapshot_inter_domain_residuals(sub: SubstrateNetwork) -> dict[str, float]:
    """Capture bw_residual for every inter-domain directed edge."""
    g = sub.graph
    residuals = {}
    for u, v, d in g.edges(data=True):
        if g.nodes[u]["domain_id"] != g.nodes[v]["domain_id"]:
            residuals[d["link_id"]] = d["bw_residual"]
    return residuals


def _snapshot_all_residuals(sub: SubstrateNetwork) -> dict:
    """Capture full residual state for exact restoration check."""
    g = sub.graph
    nodes = {n: (d["cpu_residual"], d["ram_residual"]) for n, d in g.nodes(data=True)}
    edges = {d["link_id"]: d["bw_residual"] for _, _, d in g.edges(data=True)}
    return {"nodes": nodes, "edges": edges}


class TestP3ColocatedInvariance:
    """Co-located slices must not enter the cross-domain path."""

    def test_inter_domain_demand_empty(self, colocated_slice):
        """Partition [0, 0] produces zero inter-domain demand."""
        partition = [0, 0]
        bw_demands = [fe.bandwidth_demand for fe in colocated_slice.flow_edges]
        demand = inter_domain_demand_by_pair(partition, bw_demands)
        assert demand == {}, f"Expected empty demand, got {demand}"

    def test_state_equality_and_restore(
        self, inter_domain_substrate, colocated_slice
    ):
        """Allocate + verify + deallocate: inter-domain residuals untouched,
        intra debited by beta, full restore on deallocate."""
        sub = inter_domain_substrate
        sub.reset()

        pre_all = _snapshot_all_residuals(sub)
        pre_inter = _snapshot_inter_domain_residuals(sub)

        result = greedy_place_on_substrate(sub, colocated_slice)
        assert result.feasible, f"Greedy placement should succeed: {result.fail_reason}"
        plan = result.plan
        assert plan is not None

        # Inter-domain residuals must be byte-identical after allocation
        post_inter = _snapshot_inter_domain_residuals(sub)
        assert post_inter == pre_inter, (
            "Inter-domain residuals changed for a co-located slice!\n"
            f"Before: {pre_inter}\nAfter: {post_inter}"
        )

        # Verifier: inter_domain_hops must be 0
        verdict = verify_committed_plan(sub, plan, colocated_slice)
        assert verdict.details["inter_domain_hops"] == 0, (
            f"Expected 0 inter-domain hops, got {verdict.details['inter_domain_hops']}"
        )

        # Intra link should be debited (at least one intra link residual changed)
        post_all = _snapshot_all_residuals(sub)
        intra_changed = False
        g = sub.graph
        for link_id, pre_bw in pre_all["edges"].items():
            post_bw = post_all["edges"][link_id]
            if pre_bw != post_bw:
                u, v = None, None
                for eu, ev, ed in g.edges(data=True):
                    if ed["link_id"] == link_id:
                        u, v = eu, ev
                        break
                assert g.nodes[u]["domain_id"] == g.nodes[v]["domain_id"], (
                    f"BW changed on inter-domain link {link_id}!"
                )
                intra_changed = True

        # Deallocate and verify exact restoration
        sub.deallocate(plan, colocated_slice)
        post_dealloc = _snapshot_all_residuals(sub)

        for node_id, (pre_cpu, pre_ram) in pre_all["nodes"].items():
            post_cpu, post_ram = post_dealloc["nodes"][node_id]
            assert pre_cpu == post_cpu, f"Node {node_id} CPU not restored"
            assert pre_ram == post_ram, f"Node {node_id} RAM not restored"

        for link_id, pre_bw in pre_all["edges"].items():
            post_bw = post_dealloc["edges"][link_id]
            assert pre_bw == post_bw, f"Link {link_id} BW not restored"
