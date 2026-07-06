"""Intra-domain flow routing via Yen's k-shortest paths.

After VNF placement, routes each intra-domain flow using k-shortest paths
with two-stage selection:
  1. Hard filter: all links on path must have residual BW >= flow demand.
  2. Hard filter: cumulative propagation delay <= remaining delay budget.
  3. Cost selection: min Sigma -log(residual_bw / capacity) over path links.

The log-cost form is standard in network optimization (Kelly's network utility
framework, used in MPTCP and traffic engineering). It's numerically stable
and penalizes congested links smoothly without the division-by-near-zero
pathology of inverse-residual formulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

import networkx as nx

from orion.substrate.graph_model import SubstrateNetwork


@dataclass
class RouteResult:
    """Result of routing a single intra-domain flow."""

    feasible: bool
    path_links: list[str]
    path_nodes: list[str]
    propagation_delay: float
    cost: float


class RoutingSelector:
    """Selects the best BW-feasible, delay-feasible path from k candidates."""

    def __init__(self, mode: str = "min_cost") -> None:
        if mode not in ("min_cost", "min_delay", "min_hops"):
            raise ValueError(f"Unknown routing mode: {mode}")
        self.mode = mode

    def score(
        self,
        path_nodes: list[str],
        substrate: SubstrateNetwork,
    ) -> float:
        """Score a candidate path (lower is better).

        min_cost: Sigma -log(residual_bw / capacity) — Kelly's log-cost.
        min_delay: total propagation delay.
        min_hops: number of links in the path.
        """
        g = substrate.graph
        if self.mode == "min_hops":
            return float(len(path_nodes) - 1)
        elif self.mode == "min_delay":
            return sum(
                g.edges[path_nodes[i], path_nodes[i + 1]]["propagation_delay"]
                for i in range(len(path_nodes) - 1)
            )
        else:
            cost = 0.0
            for i in range(len(path_nodes) - 1):
                ed = g.edges[path_nodes[i], path_nodes[i + 1]]
                cap = ed["bandwidth_capacity"]
                res = ed["bw_residual"]
                ratio = res / cap if cap > 0 else 1e-6
                ratio = max(ratio, 1e-6)
                cost += -math.log(ratio)
            return cost


def route_flow(
    substrate: SubstrateNetwork,
    src_node: str,
    dst_node: str,
    bw_demand: float,
    delay_budget: float,
    domain_node_ids: list[str],
    k: int = 3,
    selector: RoutingSelector | None = None,
) -> RouteResult:
    """Route a single flow within a domain subgraph.

    Uses Yen's k-shortest paths by propagation delay, then applies BW and
    delay feasibility filters, and selects the best path by cost.

    Args:
        substrate: Current substrate state.
        src_node: Source node ID (where the upstream VNF is placed).
        dst_node: Destination node ID (where the downstream VNF is placed).
        bw_demand: Minimum bandwidth required on all path links (Mbps).
        delay_budget: Maximum allowed cumulative propagation delay (ms).
        domain_node_ids: Nodes in this domain (to restrict routing).
        k: Number of shortest paths to consider.
        selector: Path scoring strategy. Defaults to min_cost.

    Returns:
        RouteResult with feasibility, path, delay, and cost.
    """
    if selector is None:
        selector = RoutingSelector("min_cost")

    if src_node == dst_node:
        return RouteResult(
            feasible=True, path_links=[], path_nodes=[src_node],
            propagation_delay=0.0, cost=0.0,
        )

    # Build domain subgraph for routing (directed)
    domain_set = set(domain_node_ids)
    subgraph = substrate.graph.subgraph(domain_set)

    # Check basic connectivity
    if not nx.has_path(subgraph, src_node, dst_node):
        return RouteResult(
            feasible=False, path_links=[], path_nodes=[],
            propagation_delay=0.0, cost=float("inf"),
        )

    try:
        k_paths = islice(
            nx.shortest_simple_paths(subgraph, src_node, dst_node, weight="propagation_delay"), k
        )
    except nx.NetworkXNoPath:
        return RouteResult(
            feasible=False, path_links=[], path_nodes=[],
            propagation_delay=0.0, cost=float("inf"),
        )

    best_result: RouteResult | None = None
    best_score = float("inf")

    for path_nodes in k_paths:

        # Check BW feasibility on all links
        bw_feasible = True
        total_delay = 0.0
        path_links: list[str] = []

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            ed = substrate.graph.edges[u, v]
            if ed["bw_residual"] < bw_demand:
                bw_feasible = False
                break
            total_delay += ed["propagation_delay"]
            path_links.append(ed["link_id"])

        if not bw_feasible:
            continue

        # Delay budget check
        if total_delay > delay_budget:
            continue

        # Score this candidate
        score = selector.score(path_nodes, substrate)
        if score < best_score:
            best_score = score
            best_result = RouteResult(
                feasible=True,
                path_links=path_links,
                path_nodes=list(path_nodes),
                propagation_delay=total_delay,
                cost=score,
            )

    if best_result is None:
        return RouteResult(
            feasible=False, path_links=[], path_nodes=[],
            propagation_delay=0.0, cost=float("inf"),
        )

    return best_result


def route_cross_domain_flow(
    substrate: SubstrateNetwork,
    src_node: str,
    dst_node: str,
    bw_demand: float,
    delay_budget: float,
    k: int = 3,
    selector: RoutingSelector | None = None,
) -> RouteResult:
    """Route a flow on the full substrate graph (cross-domain, multi-hop).

    Unlike route_flow which restricts to a single domain's subgraph, this
    operates on the entire directed substrate graph. A (0,2) flow on a
    0↔1↔2 chain will route through domain 1's internal links and the
    0-1, 1-2 inter-domain edges.

    Args:
        substrate: Current substrate state (full graph).
        src_node: Source node ID (upstream VNF placement).
        dst_node: Destination node ID (downstream VNF placement).
        bw_demand: Minimum bandwidth on every edge of the path (Mbps).
        delay_budget: Maximum cumulative propagation delay (ms).
        k: Number of shortest paths to consider.
        selector: Path scoring strategy. Defaults to min_cost.

    Returns:
        RouteResult with the multi-hop path including inter-domain edges.
    """
    if selector is None:
        selector = RoutingSelector("min_cost")

    if src_node == dst_node:
        return RouteResult(
            feasible=True, path_links=[], path_nodes=[src_node],
            propagation_delay=0.0, cost=0.0,
        )

    g = substrate.graph

    if not nx.has_path(g, src_node, dst_node):
        return RouteResult(
            feasible=False, path_links=[], path_nodes=[],
            propagation_delay=0.0, cost=float("inf"),
        )

    try:
        k_paths = islice(
            nx.shortest_simple_paths(g, src_node, dst_node, weight="propagation_delay"), k
        )
    except nx.NetworkXNoPath:
        return RouteResult(
            feasible=False, path_links=[], path_nodes=[],
            propagation_delay=0.0, cost=float("inf"),
        )

    best_result: RouteResult | None = None
    best_score = float("inf")

    for path_nodes in k_paths:

        bw_feasible = True
        total_delay = 0.0
        path_links: list[str] = []

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            ed = g.edges[u, v]
            if ed["bw_residual"] < bw_demand:
                bw_feasible = False
                break
            total_delay += ed["propagation_delay"]
            path_links.append(ed["link_id"])

        if not bw_feasible:
            continue

        if total_delay > delay_budget:
            continue

        score = selector.score(path_nodes, substrate)
        if score < best_score:
            best_score = score
            best_result = RouteResult(
                feasible=True,
                path_links=path_links,
                path_nodes=list(path_nodes),
                propagation_delay=total_delay,
                cost=score,
            )

    if best_result is None:
        return RouteResult(
            feasible=False, path_links=[], path_nodes=[],
            propagation_delay=0.0, cost=float("inf"),
        )

    return best_result


def allocate_route_bw(
    substrate: SubstrateNetwork,
    path_links: list[str],
    bw_demand: float,
) -> None:
    """Temporarily allocate bandwidth on route links (within a fragment).

    This is used during interleaved place-then-route to update edge residuals
    before placing the next VNF. The allocation is NOT committed to the
    substrate's active slice registry — that happens in Phase 5 via
    SubstrateNetwork.allocate().

    Args:
        substrate: Substrate to update in-place.
        path_links: Ordered link IDs forming the route.
        bw_demand: Bandwidth to reserve on each link.
    """
    for link_id in path_links:
        for _, _, d in substrate.graph.edges(data=True):
            if d["link_id"] == link_id:
                d["bw_residual"] -= bw_demand
                break


def deallocate_route_bw(
    substrate: SubstrateNetwork,
    path_links: list[str],
    bw_demand: float,
) -> None:
    """Undo a temporary BW allocation (on rollback / retry)."""
    for link_id in path_links:
        for _, _, d in substrate.graph.edges(data=True):
            if d["link_id"] == link_id:
                d["bw_residual"] += bw_demand
                break
