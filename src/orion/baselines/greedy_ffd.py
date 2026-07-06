"""Tier-Aware First-Fit Decreasing (FFD) baseline.

Deterministic heuristic from `IMPLEMENTATION_PLAN.md` §1 (Greedy Baseline
Specification). Used in two distinct roles, kept as two separate entry points:

1. `compute_cost_greedy(substrate, slice_req)` — runs FFD on a *copy* of the
   live substrate to produce `Cost_greedy` for the LocalScore reward term
   (v6.2 Eq. 10). Does not mutate the input substrate. (Choice B1.)

2. `greedy_place_on_substrate(substrate, slice_req)` — runs FFD on the *real*
   substrate and applies the allocation. Used by the full-episode greedy
   evaluation agent that runs on its own substrate alongside the learned
   policies (Choice B — eval baseline role).

Both share the underlying algorithm `_run_greedy_ffd`.

Algorithm:
    1. Order VNFs by decreasing CPU demand (RAM tiebreak, then vnf_id).
    2. For each VNF: candidates = D_f ∩ {nodes with residual CPU/RAM ≥ demand}.
       Sort by (tier_match DESC, residual_cpu DESC, domain_id ASC, node_id ASC).
       Place on first candidate, allocating exactly (c̃_f, r̃_f).
       If no candidate exists → infeasible.
    3. For each flow edge: same-node → no routing; else shortest-path by
       propagation delay over links with residual BW ≥ β. Allocate exactly β.
    4. Cost = α·Σ(cpu+ram) + γ_intra·Σ(intra_bw) + γ_inter·Σ(inter_bw).

Failure mode: returns Cost_greedy = +inf and a None plan. The reward layer
sets LocalScore=0 in that case to avoid divide-by-zero (plan §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from orion.substrate.graph_model import SubstrateNetwork
from orion.types import (
    InfrastructureTier,
    LinkType,
    PlacementPlan,
    SliceRequest,
    VNF,
)


# ── Public result types ──────────────────────────────────────────────────────


@dataclass
class GreedyResult:
    """Outcome of a single greedy FFD run."""

    feasible: bool
    cost: float  # +inf when infeasible
    plan: PlacementPlan | None  # None when infeasible
    intra_bw: float = 0.0
    inter_bw: float = 0.0
    resource_cost: float = 0.0  # α·Σ(cpu+ram) component, alpha=1
    fail_reason: str = ""


@dataclass
class GreedyConfig:
    """Cost weights for the greedy baseline.

    Defaults match the MDO reward's α=1, γ_inter=1 so Cost_greedy and
    Cost(π*) are commensurate without extra scaling.
    """

    alpha: float = 1.0
    gamma_intra: float = 1.0
    gamma_inter: float = 1.0


# ── Public entry points ──────────────────────────────────────────────────────


def compute_cost_greedy(
    substrate: SubstrateNetwork,
    slice_req: SliceRequest,
    config: GreedyConfig | None = None,
) -> float:
    """Compute Cost_greedy for the LocalScore reward term.

    Runs FFD without mutating `substrate` (the algorithm tracks running
    residuals internally). Returns +inf if the slice is infeasible under
    greedy — the reward layer must check for inf and set LocalScore = 0.
    """
    result = _run_greedy_ffd(substrate, slice_req, config or GreedyConfig())
    return result.cost


def greedy_place_on_substrate(
    substrate: SubstrateNetwork,
    slice_req: SliceRequest,
    config: GreedyConfig | None = None,
) -> GreedyResult:
    """Run FFD on the *real* substrate and apply the allocation.

    Used by the full-episode greedy evaluation agent (its own substrate,
    own lifecycle). On success, the substrate's residuals are decremented
    and the PlacementPlan is registered via substrate.allocate(); the caller
    is responsible for deallocating on slice departure.
    On failure, no mutation occurs.
    """
    result = _run_greedy_ffd(substrate, slice_req, config or GreedyConfig())
    if result.feasible and result.plan is not None:
        substrate.allocate(result.plan, slice_req)
    return result


# ── Internal: the algorithm ──────────────────────────────────────────────────


@dataclass
class _PlacementState:
    """Mutable working state during a single FFD run.

    `running_cpu` / `running_ram` / `running_bw` track *would-be-residual*
    deltas WITHOUT touching the real substrate. Subsequent placement
    decisions consult these to ensure consistent residual accounting.
    The caller decides whether to apply the allocation via substrate.allocate().
    """

    vnf_placements: dict[str, str] = field(default_factory=dict)
    cpu_allocations: dict[str, float] = field(default_factory=dict)
    ram_allocations: dict[str, float] = field(default_factory=dict)
    flow_routes: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    bw_allocations: dict[tuple[str, str], dict[str, float]] = field(
        default_factory=dict
    )
    intra_bw_cost: float = 0.0
    inter_bw_cost: float = 0.0
    resource_cost: float = 0.0
    running_cpu: dict[str, float] = field(default_factory=dict)
    running_ram: dict[str, float] = field(default_factory=dict)
    running_bw: dict[str, float] = field(default_factory=dict)

    def cpu_after(self, substrate: SubstrateNetwork, node_id: str) -> float:
        if node_id in self.running_cpu:
            return self.running_cpu[node_id]
        return float(substrate.graph.nodes[node_id]["cpu_residual"])

    def ram_after(self, substrate: SubstrateNetwork, node_id: str) -> float:
        if node_id in self.running_ram:
            return self.running_ram[node_id]
        return float(substrate.graph.nodes[node_id]["ram_residual"])

    def bw_after(self, substrate: SubstrateNetwork, link_id: str, u: str, v: str) -> float:
        if link_id in self.running_bw:
            return self.running_bw[link_id]
        return float(substrate.graph[u][v]["bw_residual"])


def _run_greedy_ffd(
    substrate: SubstrateNetwork,
    slice_req: SliceRequest,
    config: GreedyConfig,
) -> GreedyResult:
    """Core FFD algorithm. Does NOT mutate `substrate` — tracks running
    residuals internally. Callers that want to apply the allocation must
    invoke substrate.allocate(result.plan, slice_req) themselves.
    """
    state = _PlacementState()

    # Step 1: order VNFs by decreasing CPU, then decreasing RAM, then vnf_id.
    ordered_vnfs = sorted(
        slice_req.vnfs,
        key=lambda f: (-f.cpu_demand, -f.ram_demand, f.vnf_id),
    )

    # Step 2: place each VNF
    for vnf in ordered_vnfs:
        node_id = _select_node(substrate, vnf, state)
        if node_id is None:
            return GreedyResult(
                feasible=False,
                cost=float("inf"),
                plan=None,
                fail_reason=f"no feasible node for VNF {vnf.vnf_id}",
            )

        state.running_cpu[node_id] = (
            state.cpu_after(substrate, node_id) - vnf.cpu_demand
        )
        state.running_ram[node_id] = (
            state.ram_after(substrate, node_id) - vnf.ram_demand
        )

        state.vnf_placements[vnf.vnf_id] = node_id
        state.cpu_allocations[vnf.vnf_id] = vnf.cpu_demand
        state.ram_allocations[vnf.vnf_id] = vnf.ram_demand
        state.resource_cost += vnf.cpu_demand + vnf.ram_demand

    # Step 3: route each flow edge
    for flow in slice_req.flow_edges:
        src_node = state.vnf_placements[flow.source_vnf]
        dst_node = state.vnf_placements[flow.target_vnf]

        if src_node == dst_node:
            state.flow_routes[(flow.source_vnf, flow.target_vnf)] = []
            state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = {}
            continue

        link_ids = _shortest_bw_feasible_path(
            substrate, src_node, dst_node, flow.bandwidth_demand, state
        )
        if link_ids is None:
            return GreedyResult(
                feasible=False,
                cost=float("inf"),
                plan=None,
                fail_reason=(
                    f"no BW-feasible path from {flow.source_vnf} "
                    f"({src_node}) to {flow.target_vnf} ({dst_node})"
                ),
            )

        per_link_bw: dict[str, float] = {}
        for link_id in link_ids:
            u, v = _link_endpoints(substrate, link_id)
            state.running_bw[link_id] = (
                state.bw_after(substrate, link_id, u, v) - flow.bandwidth_demand
            )
            per_link_bw[link_id] = flow.bandwidth_demand
            link_type = substrate.graph[u][v]["link_type"]
            if link_type == LinkType.INTER.value or link_type == LinkType.INTER:
                state.inter_bw_cost += flow.bandwidth_demand
            else:
                state.intra_bw_cost += flow.bandwidth_demand

        state.flow_routes[(flow.source_vnf, flow.target_vnf)] = link_ids
        state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = per_link_bw

    # Step 4: aggregate cost
    total_cost = (
        config.alpha * state.resource_cost
        + config.gamma_intra * state.intra_bw_cost
        + config.gamma_inter * state.inter_bw_cost
    )

    plan = PlacementPlan(
        plan_id=f"{slice_req.request_id}_greedy",
        vnf_placements=state.vnf_placements,
        cpu_allocations=state.cpu_allocations,
        ram_allocations=state.ram_allocations,
        flow_routes=state.flow_routes,
        bw_allocations=state.bw_allocations,
        is_structurally_valid=True,
        source="greedy",
    )
    return GreedyResult(
        feasible=True,
        cost=total_cost,
        plan=plan,
        intra_bw=state.intra_bw_cost,
        inter_bw=state.inter_bw_cost,
        resource_cost=state.resource_cost,
    )


# ── Internal: helpers ────────────────────────────────────────────────────────


def _select_node(
    substrate: SubstrateNetwork,
    vnf: VNF,
    state: _PlacementState,
) -> str | None:
    """Choose the best feasible node for `vnf` per the FFD sort key.

    Sort key (descending priority): (tier_match DESC, residual_cpu DESC,
    domain_id ASC, node_id ASC). C8 placement rule = vnf.permitted_nodes.
    Residual is consulted via `state` so prior VNFs in the same run see
    consistent capacity, without mutating the substrate.
    """
    g = substrate.graph
    permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None

    candidates: list[tuple[int, float, int, str]] = []  # sort key + node_id
    required_tier = _required_tier(vnf, substrate)

    for node_id, d in g.nodes(data=True):
        if permitted is not None and node_id not in permitted:
            continue
        cpu_avail = state.cpu_after(substrate, node_id)
        ram_avail = state.ram_after(substrate, node_id)
        if cpu_avail < vnf.cpu_demand:
            continue
        if ram_avail < vnf.ram_demand:
            continue

        tier_match = 1 if (required_tier is not None and d["tier"] == required_tier) else 0
        candidates.append(
            (
                -tier_match,            # DESC tier_match
                -cpu_avail,             # DESC residual_cpu (effective)
                d["domain_id"],         # ASC domain_id
                node_id,                # ASC node_id
            )
        )

    if not candidates:
        return None

    candidates.sort()
    # Return the node_id of the best candidate.
    return candidates[0][3]


def _required_tier(vnf: VNF, substrate: SubstrateNetwork) -> str | None:
    """Infer the VNF's required tier from its permitted_nodes (modal tier).

    If permitted_nodes lists only nodes of one tier, that tier is "required."
    Otherwise return None (no tier preference for tier_match scoring).
    """
    if not vnf.permitted_nodes:
        return None
    tiers = {substrate.graph.nodes[n]["tier"] for n in vnf.permitted_nodes
             if n in substrate.graph.nodes}
    if len(tiers) == 1:
        return next(iter(tiers))
    return None


def _shortest_bw_feasible_path(
    substrate: SubstrateNetwork,
    src: str,
    dst: str,
    required_bw: float,
    state: _PlacementState,
) -> list[str] | None:
    """Shortest propagation-delay path with residual BW ≥ required_bw.

    BW availability consults `state.running_bw` so already-routed flows in
    the same run are accounted for without mutating the substrate.
    Returns the ordered list of link_ids along the path, or None if no path.
    """
    g = substrate.graph

    def edge_ok(u: str, v: str) -> bool:
        link_id = g[u][v]["link_id"]
        return state.bw_after(substrate, link_id, u, v) >= required_bw

    sub = nx.subgraph_view(g, filter_edge=edge_ok)
    try:
        node_path = nx.shortest_path(sub, src, dst, weight="propagation_delay")
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None

    link_ids: list[str] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        link_ids.append(g[u][v]["link_id"])
    return link_ids


def _link_endpoints(substrate: SubstrateNetwork, link_id: str) -> tuple[str, str]:
    """Resolve a link_id to its (source, target) node IDs.

    Linear scan — substrate edges aren't indexed by link_id today. Acceptable
    here since FFD touches few edges per slice.
    """
    for u, v, d in substrate.graph.edges(data=True):
        if d["link_id"] == link_id:
            return u, v
    raise KeyError(f"link_id {link_id!r} not found in substrate")


# Re-export the tier symbol so consumers don't need a second import.
__all__ = [
    "GreedyConfig",
    "GreedyResult",
    "InfrastructureTier",
    "compute_cost_greedy",
    "greedy_place_on_substrate",
]
