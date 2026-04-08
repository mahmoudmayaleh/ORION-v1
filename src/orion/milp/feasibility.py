"""Feasibility checker for placement plans.

  Inference mode — binary variables fixed by the RL actors' selection.
  - check():            C2 (CPU), C3 (RAM), C5 (link BW), C5b (throughput), C7 (delay)
  - check_structural(): C1, C4, C6, C8 (guaranteed upstream by Agent B)
"""

from __future__ import annotations

from collections import defaultdict

from orion.substrate.graph_model import SubstrateNetwork
from orion.types import FeasibilityResult, PlacementPlan, SliceRequest

_EPS = 1e-6


class FeasibilityChecker:
    def __init__(self, substrate: SubstrateNetwork) -> None:
        self.substrate = substrate

    def check(
        self,
        plan: PlacementPlan,
        request: SliceRequest,
    ) -> FeasibilityResult:
        """Resource feasibility: C2, C3, C5, C5b, C7."""
        violations: list[str] = []
        details: dict[str, float] = {}

        # C2 + C3: per-node aggregate allocation <= residual capacity
        node_cpu: dict[str, float] = defaultdict(float)
        node_ram: dict[str, float] = defaultdict(float)
        for fid, n_id in plan.vnf_placements.items():
            node_cpu[n_id] += plan.cpu_allocations.get(fid, 0.0)
            node_ram[n_id] += plan.ram_allocations.get(fid, 0.0)

        for n_id, alloc_cpu in node_cpu.items():
            res_cpu = self.substrate.get_residual_cpu(n_id)
            if alloc_cpu > res_cpu + _EPS:
                key = f"C2_cpu_{n_id}"
                violations.append(key)
                details[key] = alloc_cpu - res_cpu

        for n_id, alloc_ram in node_ram.items():
            res_ram = self.substrate.get_residual_ram(n_id)
            if alloc_ram > res_ram + _EPS:
                key = f"C3_ram_{n_id}"
                violations.append(key)
                details[key] = alloc_ram - res_ram

        # C5: per-link bandwidth <= residual
        link_bw: dict[str, float] = defaultdict(float)
        for per_link in plan.bw_allocations.values():
            for l_id, bw in per_link.items():
                link_bw[l_id] += bw

        for l_id, alloc_bw in link_bw.items():
            try:
                res_bw = self.substrate.get_residual_bw(l_id)
            except KeyError:
                key = f"C5_unknown_link_{l_id}"
                violations.append(key)
                details[key] = alloc_bw
                continue
            if alloc_bw > res_bw + _EPS:
                key = f"C5_bw_{l_id}"
                violations.append(key)
                details[key] = alloc_bw - res_bw

        # C5b: per-flow throughput >= beta_min_s
        beta_min = request.qos.min_throughput
        for edge in request.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            per_link = plan.bw_allocations.get(flow_key, {})
            total_bw = sum(per_link.values())
            if total_bw < beta_min - _EPS:
                key = f"C5b_{edge.source_vnf}_{edge.target_vnf}"
                violations.append(key)
                details[key] = beta_min - total_bw

        # C7: end-to-end delay budget
        total_delay = 0.0
        for f in request.vnfs:
            n_id = plan.vnf_placements.get(f.vnf_id)
            if n_id is not None:
                total_delay += self.substrate.graph.nodes[n_id]["processing_delay"]

        for edge in request.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            for l_id in plan.flow_routes.get(flow_key, []):
                try:
                    total_delay += self.substrate.get_link_attrs(l_id).propagation_delay
                except KeyError:
                    pass

        if total_delay > request.qos.max_e2e_delay + _EPS:
            violations.append("C7_delay")
            details["C7_delay"] = total_delay - request.qos.max_e2e_delay

        return FeasibilityResult(
            is_feasible=len(violations) == 0,
            violated_constraints=violations,
            violation_details=details,
        )

    def check_structural(
        self,
        plan: PlacementPlan,
        request: SliceRequest,
    ) -> FeasibilityResult:
        """Structural feasibility: C1, C4, C6, C8 (no LP solve)."""
        violations: list[str] = []
        details: dict[str, float] = {}
        vnf_map = {f.vnf_id: f for f in request.vnfs}

        # C1: each VNF placed on exactly one node
        for f in request.vnfs:
            if f.vnf_id not in plan.vnf_placements:
                key = f"C1_missing_{f.vnf_id}"
                violations.append(key)
                details[key] = 1.0

        # C8: VNF placed on a permitted node
        for fid, n_id in plan.vnf_placements.items():
            if fid in vnf_map and n_id not in vnf_map[fid].permitted_nodes:
                key = f"C8_{fid}"
                violations.append(key)
                details[key] = 1.0

        # C4: allocation >= VNF minimum demands
        for f in request.vnfs:
            cpu_alloc = plan.cpu_allocations.get(f.vnf_id, 0.0)
            ram_alloc = plan.ram_allocations.get(f.vnf_id, 0.0)
            if cpu_alloc < f.cpu_demand - _EPS:
                key = f"C4_cpu_{f.vnf_id}"
                violations.append(key)
                details[key] = f.cpu_demand - cpu_alloc
            if ram_alloc < f.ram_demand - _EPS:
                key = f"C4_ram_{f.vnf_id}"
                violations.append(key)
                details[key] = f.ram_demand - ram_alloc

        # C6: route is a valid directed path from placed(f) to placed(f')
        for edge in request.flow_edges:
            flow_key = (edge.source_vnf, edge.target_vnf)
            src_node = plan.vnf_placements.get(edge.source_vnf)
            dst_node = plan.vnf_placements.get(edge.target_vnf)
            route = plan.flow_routes.get(flow_key, [])

            if src_node is None or dst_node is None:
                continue

            if src_node == dst_node:
                if route:
                    key = f"C6_colocated_{edge.source_vnf}_{edge.target_vnf}"
                    violations.append(key)
                    details[key] = float(len(route))
            else:
                if not _is_valid_path(self.substrate, src_node, dst_node, route):
                    key = f"C6_path_{edge.source_vnf}_{edge.target_vnf}"
                    violations.append(key)
                    details[key] = 1.0

        return FeasibilityResult(
            is_feasible=len(violations) == 0,
            violated_constraints=violations,
            violation_details=details,
        )


def _is_valid_path(
    substrate: SubstrateNetwork,
    src_node: str,
    dst_node: str,
    link_ids: list[str],
) -> bool:
    """Check that link_ids form a connected directed path from src_node to dst_node.

    Args:
        substrate: Substrate providing edge structure.
        src_node: Expected path start node.
        dst_node: Expected path end node.
        link_ids: List of directed link IDs.

    Returns:
        True if the links form a valid directed path from src_node to dst_node.
    """
    if not link_ids:
        return False

    adjacency: dict[str, str] = {}
    link_set = set(link_ids)
    for u, v, d in substrate.graph.edges(data=True):
        if d["link_id"] in link_set:
            adjacency[u] = v

    current = src_node
    visited: set[str] = set()
    while current in adjacency:
        if current in visited:
            return False  # cycle
        visited.add(current)
        current = adjacency[current]

    return current == dst_node
