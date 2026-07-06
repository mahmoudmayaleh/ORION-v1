"""Co-location-first plan builders for memory experiments.

Three variants sharing the same interface as _run_greedy_ffd:
  builder(substrate, slice_req, config) -> GreedyResult

1. colocation_ffd: Try single-domain first, fall back to FFD.
2. routability_aware_colocation_ffd: Like colocation_ffd but when falling
   back to cross-domain, orders candidate domains by inter-domain BW
   residual and checks routability before committing a split. This is the
   strong static baseline for memory experiments.
3. pure_colocation: Single-domain only, no fallback. Rejects if no single
   domain can host the whole chain.
"""

from __future__ import annotations

import copy

import networkx as nx

from orion.actors.routing import route_cross_domain_flow, allocate_route_bw, deallocate_route_bw
from orion.baselines.greedy_ffd import (
    GreedyConfig, GreedyResult, _PlacementState, _run_greedy_ffd,
    _shortest_bw_feasible_path, _link_endpoints, _required_tier,
)
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import LinkType, PlacementPlan, SliceRequest, VNF


# ── Shared helpers ──────────────────────────────────────────────────────────


def _select_node_in_domain(substrate, vnf, state, domain_nodes):
    """Best-fit node selection within a specific domain."""
    g = substrate.graph
    permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None
    required_tier = _required_tier(vnf, substrate)
    candidates = []

    for node_id in domain_nodes:
        d = g.nodes[node_id]
        if permitted is not None and node_id not in permitted:
            continue
        cpu_avail = state.cpu_after(substrate, node_id)
        ram_avail = state.ram_after(substrate, node_id)
        if cpu_avail < vnf.cpu_demand or ram_avail < vnf.ram_demand:
            continue
        tier_match = 1 if (required_tier and d["tier"] == required_tier) else 0
        residual_after = cpu_avail - vnf.cpu_demand
        candidates.append((-tier_match, residual_after, node_id))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _try_single_domain(substrate, slice_req, config, domain_id, domain_nodes):
    """Try placing all VNFs in one domain. Returns GreedyResult or None."""
    state = _PlacementState()

    ordered_vnfs = sorted(
        slice_req.vnfs,
        key=lambda f: (-f.cpu_demand, -f.ram_demand, f.vnf_id),
    )

    for vnf in ordered_vnfs:
        node_id = _select_node_in_domain(substrate, vnf, state, domain_nodes)
        if node_id is None:
            return None
        state.running_cpu[node_id] = state.cpu_after(substrate, node_id) - vnf.cpu_demand
        state.running_ram[node_id] = state.ram_after(substrate, node_id) - vnf.ram_demand
        state.vnf_placements[vnf.vnf_id] = node_id
        state.cpu_allocations[vnf.vnf_id] = vnf.cpu_demand
        state.ram_allocations[vnf.vnf_id] = vnf.ram_demand
        state.resource_cost += vnf.cpu_demand + vnf.ram_demand

    # Route all flows (all intra-domain)
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
            return None
        per_link_bw = {}
        for lid in link_ids:
            u, v = _link_endpoints(substrate, lid)
            state.running_bw[lid] = state.bw_after(substrate, lid, u, v) - flow.bandwidth_demand
            per_link_bw[lid] = flow.bandwidth_demand
            state.intra_bw_cost += flow.bandwidth_demand
        state.flow_routes[(flow.source_vnf, flow.target_vnf)] = link_ids
        state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = per_link_bw

    total_cost = (config.alpha * state.resource_cost
                  + config.gamma_intra * state.intra_bw_cost)
    plan = PlacementPlan(
        plan_id=f"{slice_req.request_id}_coloc",
        vnf_placements=state.vnf_placements,
        cpu_allocations=state.cpu_allocations,
        ram_allocations=state.ram_allocations,
        flow_routes=state.flow_routes,
        bw_allocations=state.bw_allocations,
        is_structurally_valid=True, source="colocation",
    )
    return GreedyResult(feasible=True, cost=total_cost, plan=plan,
                        intra_bw=state.intra_bw_cost, inter_bw=0.0,
                        resource_cost=state.resource_cost)


def _domain_cpu_ranking(substrate):
    """Rank domains by total residual CPU (descending)."""
    g = substrate.graph
    domain_cpu = {}
    domain_nodes = {}
    for nid, d in g.nodes(data=True):
        dom = d.get("domain_id", -1)
        if dom < 0:
            continue
        domain_cpu[dom] = domain_cpu.get(dom, 0.0) + float(d["cpu_residual"])
        domain_nodes.setdefault(dom, set()).add(nid)
    return sorted(domain_cpu.keys(), key=lambda d: -domain_cpu[d]), domain_nodes


# ── Variant 1: Co-location with FFD fallback ───────────────────────────────


def colocation_ffd(substrate, slice_req, config=None):
    """Co-location-first: try single-domain, fall back to FFD."""
    config = config or GreedyConfig()
    ranked_domains, domain_nodes = _domain_cpu_ranking(substrate)

    for dom in ranked_domains:
        result = _try_single_domain(
            substrate, slice_req, config, dom, sorted(domain_nodes[dom])
        )
        if result is not None:
            return result

    return _run_greedy_ffd(substrate, slice_req, config)


# ── Variant 2: Routability-aware co-location (strong baseline) ─────────────


def routability_aware_colocation_ffd(substrate, slice_req, config=None):
    """Routability-aware co-location: the strong static baseline.

    1. Try single-domain placement (best-first by CPU residual)
    2. If no single domain works, fall back to cross-domain with BW awareness:
       a. For each VNF, prefer domains already used by this slice
       b. Among new-domain candidates, rank by inter-domain BW residual
          to the domains already in use
       c. After placing all VNFs, verify cross-domain flow routability
          BEFORE committing (rejects early if routing fails)
    """
    config = config or GreedyConfig()
    ranked_domains, domain_node_sets = _domain_cpu_ranking(substrate)

    # Phase 1: try single-domain
    for dom in ranked_domains:
        result = _try_single_domain(
            substrate, slice_req, config, dom, sorted(domain_node_sets[dom])
        )
        if result is not None:
            return result

    # Phase 2: BW-aware cross-domain placement
    g = substrate.graph

    # Precompute inter-domain BW residuals between domain pairs
    inter_bw_residual: dict[tuple[int, int], float] = {}
    for u, v, d in g.edges(data=True):
        u_dom = g.nodes[u].get("domain_id", -1)
        v_dom = g.nodes[v].get("domain_id", -1)
        if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
            pair = (min(u_dom, v_dom), max(u_dom, v_dom))
            inter_bw_residual[pair] = inter_bw_residual.get(pair, 0.0) + float(d["bw_residual"])

    state = _PlacementState()
    ordered_vnfs = sorted(
        slice_req.vnfs,
        key=lambda f: (-f.cpu_demand, -f.ram_demand, f.vnf_id),
    )

    placed_domains: set[int] = set()

    for vnf in ordered_vnfs:
        node_id = _select_node_bw_aware(
            substrate, vnf, state, domain_node_sets, placed_domains, inter_bw_residual
        )
        if node_id is None:
            return GreedyResult(feasible=False, cost=float("inf"), plan=None,
                                fail_reason=f"no feasible node for VNF {vnf.vnf_id}")

        state.running_cpu[node_id] = state.cpu_after(substrate, node_id) - vnf.cpu_demand
        state.running_ram[node_id] = state.ram_after(substrate, node_id) - vnf.ram_demand
        state.vnf_placements[vnf.vnf_id] = node_id
        state.cpu_allocations[vnf.vnf_id] = vnf.cpu_demand
        state.ram_allocations[vnf.vnf_id] = vnf.ram_demand
        state.resource_cost += vnf.cpu_demand + vnf.ram_demand
        placed_domains.add(g.nodes[node_id]["domain_id"])

    # Route intra-domain flows
    for flow in slice_req.flow_edges:
        src_node = state.vnf_placements[flow.source_vnf]
        dst_node = state.vnf_placements[flow.target_vnf]
        src_dom = g.nodes[src_node]["domain_id"]
        dst_dom = g.nodes[dst_node]["domain_id"]

        if src_node == dst_node:
            state.flow_routes[(flow.source_vnf, flow.target_vnf)] = []
            state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = {}
            continue

        if src_dom == dst_dom:
            domain_nids = sorted(domain_node_sets[src_dom])
            link_ids = _shortest_bw_feasible_path(
                substrate, src_node, dst_node, flow.bandwidth_demand, state
            )
            if link_ids is None:
                return GreedyResult(feasible=False, cost=float("inf"), plan=None,
                                    fail_reason=f"no intra-domain BW path for {flow.source_vnf}->{flow.target_vnf}")
            per_link_bw = {}
            for lid in link_ids:
                u, v = _link_endpoints(substrate, lid)
                state.running_bw[lid] = state.bw_after(substrate, lid, u, v) - flow.bandwidth_demand
                per_link_bw[lid] = flow.bandwidth_demand
                state.intra_bw_cost += flow.bandwidth_demand
            state.flow_routes[(flow.source_vnf, flow.target_vnf)] = link_ids
            state.bw_allocations[(flow.source_vnf, flow.target_vnf)] = per_link_bw

    # Verify cross-domain flow routability BEFORE committing
    sub_check = copy.deepcopy(substrate)
    cross_routes = {}
    cross_bw = {}
    cross_feasible = True

    for flow in slice_req.flow_edges:
        src_node = state.vnf_placements[flow.source_vnf]
        dst_node = state.vnf_placements[flow.target_vnf]
        if src_node == dst_node:
            continue
        src_dom = g.nodes[src_node]["domain_id"]
        dst_dom = g.nodes[dst_node]["domain_id"]
        if src_dom == dst_dom:
            continue

        result = route_cross_domain_flow(
            sub_check, src_node, dst_node,
            bw_demand=flow.bandwidth_demand,
            delay_budget=999999.0,
        )
        if not result.feasible:
            cross_feasible = False
            # Rollback
            for fk, bw in cross_bw.items():
                deallocate_route_bw(sub_check, cross_routes[fk], bw)
            break

        fk = (flow.source_vnf, flow.target_vnf)
        cross_routes[fk] = result.path_links
        cross_bw[fk] = flow.bandwidth_demand
        allocate_route_bw(sub_check, result.path_links, flow.bandwidth_demand)
        state.inter_bw_cost += flow.bandwidth_demand * len(result.path_links)

    if not cross_feasible:
        # Cross-domain routing failed — try regular FFD as last resort
        return _run_greedy_ffd(substrate, slice_req, config)

    # Commit: add cross-domain routes to the plan
    for fk, link_ids in cross_routes.items():
        state.flow_routes[fk] = link_ids
        state.bw_allocations[fk] = {lid: cross_bw[fk] for lid in link_ids}

    total_cost = (config.alpha * state.resource_cost
                  + config.gamma_intra * state.intra_bw_cost
                  + config.gamma_inter * state.inter_bw_cost)
    plan = PlacementPlan(
        plan_id=f"{slice_req.request_id}_ra_coloc",
        vnf_placements=state.vnf_placements,
        cpu_allocations=state.cpu_allocations,
        ram_allocations=state.ram_allocations,
        flow_routes=state.flow_routes,
        bw_allocations=state.bw_allocations,
        is_structurally_valid=True, source="routability_aware_colocation",
    )
    return GreedyResult(feasible=True, cost=total_cost, plan=plan,
                        intra_bw=state.intra_bw_cost, inter_bw=state.inter_bw_cost,
                        resource_cost=state.resource_cost)


def _select_node_bw_aware(substrate, vnf, state, domain_node_sets, placed_domains, inter_bw_residual):
    """Node selection with co-location preference + BW-aware cross-domain ordering.

    Priority:
      1. Nodes in already-used domains (co-location preference)
      2. Among new domains, prefer those with highest inter-domain BW residual
         to the domains already in use
      3. Within a domain, best-fit by CPU residual
    """
    g = substrate.graph
    permitted = set(vnf.permitted_nodes) if vnf.permitted_nodes else None
    required_tier = _required_tier(vnf, substrate)
    candidates = []

    for dom, node_set in domain_node_sets.items():
        for nid in node_set:
            d = g.nodes[nid]
            if permitted is not None and nid not in permitted:
                continue
            if required_tier and d["tier"] != required_tier:
                continue
            cpu_avail = state.cpu_after(substrate, nid)
            ram_avail = state.ram_after(substrate, nid)
            if cpu_avail < vnf.cpu_demand or ram_avail < vnf.ram_demand:
                continue

            # Scoring
            in_placed = 1 if dom in placed_domains else 0
            # BW to placed domains (higher = better for cross-domain)
            bw_to_placed = 0.0
            if placed_domains and dom not in placed_domains:
                for pd in placed_domains:
                    pair = (min(dom, pd), max(dom, pd))
                    bw_to_placed += inter_bw_residual.get(pair, 0.0)

            residual_after = cpu_avail - vnf.cpu_demand

            candidates.append((
                -in_placed,         # Prefer same domain (descending)
                -bw_to_placed,      # If new domain, prefer high BW to placed (descending)
                residual_after,     # Best-fit within domain (ascending)
                nid,
            ))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


# ── Variant 3: Pure co-location (no fallback) ──────────────────────────────


def pure_colocation(substrate, slice_req, config=None):
    """Pure co-location: reject if no single domain can host the chain."""
    config = config or GreedyConfig()
    ranked_domains, domain_nodes = _domain_cpu_ranking(substrate)

    for dom in ranked_domains:
        result = _try_single_domain(
            substrate, slice_req, config, dom, sorted(domain_nodes[dom])
        )
        if result is not None:
            return result

    return GreedyResult(feasible=False, cost=float("inf"), plan=None,
                        fail_reason="no single-domain placement (pure co-location)")
