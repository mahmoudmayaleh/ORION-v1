"""Deterministic best-fit domain actor for memory experiments.

Implements the same .act(substrate, fragment) interface as DomainActor but
uses best-fit placement (tightest CPU fit) and shortest-path routing. No
learnable parameters. Identical across all experiment arms — removes actor
stochasticity from the memory comparison.

Node selection: best-fit minimizes fragmentation by picking the feasible
node where the VNF leaves the least residual CPU. Tier match is a hard
constraint. Deterministic tiebreak by node_id.
"""

from __future__ import annotations

import torch

from orion.actors.routing import (
    allocate_route_bw,
    deallocate_route_bw,
    route_flow,
    RoutingSelector,
)
from orion.actors.types import DomainResponse, PlanFragment, VNFAssignment
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier


_TIER_ORDER = {
    InfrastructureTier.RAN_EDGE: 0,
    InfrastructureTier.MEC: 1,
    InfrastructureTier.REGIONAL_CLOUD: 2,
    InfrastructureTier.CENTRAL_CLOUD: 3,
}


class GreedyDomainActor:
    """Deterministic best-fit placer within a single domain.

    Sorts VNFs by decreasing CPU then RAM, picks the best-fit feasible node
    (tier match first, then tightest CPU fit), routes intra-domain flows
    on the shortest delay-feasible path. No RL signals produced.
    Identical across all experiment arms.
    """

    def __init__(self, domain_id: int, k_paths: int = 3):
        self.domain_id = domain_id
        self.k_paths = k_paths
        self.routing_selector = RoutingSelector("min_delay")

    def act(
        self,
        substrate: SubstrateNetwork,
        fragment: PlanFragment,
        deterministic: bool = True,
    ) -> DomainResponse:
        if fragment.is_empty:
            return DomainResponse.empty(self.domain_id)

        g = substrate.graph
        domain_nodes = sorted(substrate.nodes_in_domain(self.domain_id))
        if not domain_nodes:
            return DomainResponse(domain_id=self.domain_id, feasible=False)

        vnf_list = sorted(
            fragment.vnf_assignments,
            key=lambda v: (-v.cpu_demand, -v.ram_demand, v.vnf_id),
        )

        placements: dict[str, str] = {}
        node_snapshots: dict[str, tuple[float, float]] = {}
        allocated_bw: list[tuple[list[str], float]] = []
        routes: dict[tuple[str, str], list[str]] = {}
        bw_allocated: dict[tuple[str, str], float] = {}
        total_proc_delay = 0.0
        total_route_delay = 0.0
        total_resource_cost = 0.0

        intra_flow_by_target: dict[str, tuple[str, float]] = {}
        for fe in fragment.intra_flows:
            intra_flow_by_target[fe.target_vnf] = (fe.source_vnf, fe.bandwidth_demand)

        delay_remaining = fragment.delay_budget_ms

        for vnf in vnf_list:
            node_id = self._select_node(substrate, vnf, domain_nodes)
            if node_id is None:
                self._rollback(substrate, allocated_bw, node_snapshots)
                return DomainResponse(
                    domain_id=self.domain_id, feasible=False,
                )

            placements[vnf.vnf_id] = node_id

            if node_id not in node_snapshots:
                node_snapshots[node_id] = (
                    g.nodes[node_id]["cpu_residual"],
                    g.nodes[node_id]["ram_residual"],
                )

            g.nodes[node_id]["cpu_residual"] -= vnf.cpu_demand
            g.nodes[node_id]["ram_residual"] -= vnf.ram_demand

            proc_delay = (
                g.nodes[node_id]["processing_delay"] * vnf.computational_intensity
            )
            total_proc_delay += proc_delay
            total_resource_cost += vnf.cpu_demand + vnf.ram_demand

            if vnf.vnf_id in intra_flow_by_target:
                src_vnf_id, bw_demand = intra_flow_by_target[vnf.vnf_id]
                if src_vnf_id not in placements:
                    continue
                src_node = placements[src_vnf_id]
                dst_node = node_id

                result = route_flow(
                    substrate, src_node, dst_node,
                    bw_demand=bw_demand,
                    delay_budget=delay_remaining,
                    domain_node_ids=domain_nodes,
                    k=self.k_paths,
                    selector=self.routing_selector,
                )

                if not result.feasible:
                    self._rollback(substrate, allocated_bw, node_snapshots)
                    return DomainResponse(
                        domain_id=self.domain_id, feasible=False,
                    )

                flow_key = (src_vnf_id, vnf.vnf_id)
                routes[flow_key] = result.path_links
                bw_allocated[flow_key] = bw_demand
                total_route_delay += result.propagation_delay
                delay_remaining -= result.propagation_delay

                allocate_route_bw(substrate, result.path_links, bw_demand)
                allocated_bw.append((result.path_links, bw_demand))

        return DomainResponse(
            domain_id=self.domain_id,
            feasible=True,
            placements=placements,
            routes=routes,
            bw_allocated=bw_allocated,
            intra_delay=total_proc_delay + total_route_delay,
            resource_cost=total_resource_cost,
            actions=[],
            log_probs=torch.tensor([]),
            entropy=0.0,
            step_records=[],
        )

    def _select_node(
        self,
        substrate: SubstrateNetwork,
        vnf: VNFAssignment,
        domain_nodes: list[str],
    ) -> str | None:
        """Best-fit node selection: tightest CPU fit after placement.

        Minimizes fragmentation. Tier match is a hard constraint.
        Deterministic tiebreak by node_id.
        """
        g = substrate.graph
        permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None
        required_tier = vnf.required_tier
        candidates: list[tuple[int, float, str]] = []

        for node_id in domain_nodes:
            if permitted is not None and node_id not in permitted:
                continue
            d = g.nodes[node_id]
            cpu_avail = float(d["cpu_residual"])
            ram_avail = float(d["ram_residual"])
            if cpu_avail < vnf.cpu_demand or ram_avail < vnf.ram_demand:
                continue

            tier_match = 1 if d["tier"] == required_tier.value else 0
            residual_after = cpu_avail - vnf.cpu_demand
            # Best-fit: tightest CPU fit (ascending residual), tier match required
            candidates.append((-tier_match, residual_after, node_id))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def _rollback(
        self,
        substrate: SubstrateNetwork,
        allocated_bw: list[tuple[list[str], float]],
        node_snapshots: dict[str, tuple[float, float]],
    ) -> None:
        for path_links, bw in reversed(allocated_bw):
            deallocate_route_bw(substrate, path_links, bw)
        g = substrate.graph
        for node_id, (orig_cpu, orig_ram) in node_snapshots.items():
            g.nodes[node_id]["cpu_residual"] = orig_cpu
            g.nodes[node_id]["ram_residual"] = orig_ram
