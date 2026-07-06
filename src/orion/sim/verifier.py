"""Post-commit ground-truth verification of a committed PlacementPlan.

This is the **simulator-side** check that fires the hard penalty in the
reward (v6.2 Section 3.2, "ground-truth vs light-load gap"). It is distinct
from `orion.mdo.precommit_check`, which uses domain-reported values under
the light-load assumption.

The gap is real because:
  - Pre-commit (MDO side): static propagation delay only, summed from domain
    actor reports.
  - Post-commit (here): M/M/1 sojourn over the actually-allocated nodes and
    links, using current substrate residuals.

A C7 (delay) violation here that the MDO did not foresee at pre-commit IS
the load-dependent gap the paper claims to characterise. Without it, C7
would be structurally undetectable and the hard-penalty term would be dead
weight (Choice E1 + C1).

Constraints checked:
    C2  — node CPU capacity (no over-allocation)
    C3  — node RAM capacity
    C5  — per-link bandwidth capacity
    C5b — slice throughput floor (VCR-scaled β_min)
    C7  — E2E delay budget under load-dependent M/M/1 sojourn
    C9  — inter-domain hop limit
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from orion.sim.delay_model import link_sojourn, node_sojourn
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import LinkType, PlacementPlan, SliceRequest, VNF


@dataclass
class GroundTruthVerdict:
    """Result of post-commit verification.

    Attributes:
        feasible: True iff no hard constraints violated.
        violated: List of constraint codes that fired (e.g., ["C7", "C9"]).
        details: Numeric breakdown for logging / reward shaping.
            Keys (all optional):
                e2e_delay, delay_budget, delay_slack
                achieved_throughput, throughput_floor
                inter_domain_hops, hop_limit
                worst_cpu_overflow, worst_ram_overflow, worst_bw_overflow
    """

    feasible: bool
    violated: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)

    @property
    def hard_penalty_fired(self) -> bool:
        """Whether the reward's hard-penalty indicator should fire.

        Per v6.2 Eq. 9, the hard penalty is 1[C2 ∨ C3 ∨ C5b ∨ C7 violated].
        C5 and C9 do not trigger the hard penalty (they are caught by the
        MDO pre-commit / dispatch logic and would not normally slip through).
        """
        return any(c in self.violated for c in ("C2", "C3", "C5b", "C7"))


def verify_committed_plan(
    substrate: SubstrateNetwork,
    plan: PlacementPlan,
    slice_req: SliceRequest,
    max_inter_domain_hops: int = 3,
) -> GroundTruthVerdict:
    """Post-commit ground-truth check of a placed slice.

    Assumes `plan` has already been applied to `substrate` via
    `substrate.allocate(plan, slice_req)`. The verifier reads the *current*
    residual state and reasons against the *post-allocation* load.

    Args:
        substrate: The substrate AFTER allocation has been applied.
        plan: The committed placement plan.
        slice_req: The slice request that was admitted.
        max_inter_domain_hops: C9 limit (default 3 per plan §1).

    Returns:
        A GroundTruthVerdict.
    """
    violated: list[str] = []
    details: dict[str, float] = {}

    _check_node_capacities(substrate, violated, details)
    _check_link_capacities(substrate, violated, details)

    e2e = _compute_ground_truth_e2e(substrate, plan, slice_req)
    details["e2e_delay"] = e2e
    details["delay_budget"] = slice_req.qos.max_e2e_delay
    details["delay_slack"] = slice_req.qos.max_e2e_delay - e2e
    if e2e > slice_req.qos.max_e2e_delay:
        violated.append("C7")

    delivered_beta_in, c5b_violated = _check_c5b(slice_req, plan)
    details["achieved_throughput"] = delivered_beta_in
    details["throughput_floor"] = slice_req.qos.min_throughput
    if c5b_violated:
        violated.append("C5b")

    hops = _count_inter_domain_hops(substrate, plan)
    details["inter_domain_hops"] = float(hops)
    details["hop_limit"] = float(max_inter_domain_hops)
    if hops > max_inter_domain_hops:
        violated.append("C9")

    return GroundTruthVerdict(
        feasible=not violated,
        violated=violated,
        details=details,
    )


# ── Internal checks ──────────────────────────────────────────────────────────


def _check_node_capacities(
    substrate: SubstrateNetwork,
    violated: list[str],
    details: dict[str, float],
) -> None:
    """C2 (CPU) and C3 (RAM). A residual < 0 means over-allocation."""
    worst_cpu = 0.0
    worst_ram = 0.0
    cpu_overflow = False
    ram_overflow = False
    for _, d in substrate.graph.nodes(data=True):
        if d["cpu_residual"] < 0:
            cpu_overflow = True
            worst_cpu = min(worst_cpu, d["cpu_residual"])
        if d["ram_residual"] < 0:
            ram_overflow = True
            worst_ram = min(worst_ram, d["ram_residual"])
    if cpu_overflow:
        violated.append("C2")
        details["worst_cpu_overflow"] = -worst_cpu
    if ram_overflow:
        violated.append("C3")
        details["worst_ram_overflow"] = -worst_ram


def _check_link_capacities(
    substrate: SubstrateNetwork,
    violated: list[str],
    details: dict[str, float],
) -> None:
    """C5 — link bandwidth. Residual < 0 means over-allocation."""
    worst_bw = 0.0
    bw_overflow = False
    for _, _, d in substrate.graph.edges(data=True):
        if d["bw_residual"] < 0:
            bw_overflow = True
            worst_bw = min(worst_bw, d["bw_residual"])
    if bw_overflow:
        violated.append("C5")
        details["worst_bw_overflow"] = -worst_bw


def _compute_ground_truth_e2e(
    substrate: SubstrateNetwork,
    plan: PlacementPlan,
    slice_req: SliceRequest,
) -> float:
    """E2E delay under load-dependent M/M/1 sojourn (C1 + E1).

    Sums per-VNF node sojourn (intensity × base + 1/(μ−λ)) plus per-link
    sojourn for each routed flow edge. If any element is saturated, returns
    +inf, which will trip C7 cleanly.
    """
    g = substrate.graph
    vnf_by_id: dict[str, VNF] = {v.vnf_id: v for v in slice_req.vnfs}

    total = 0.0

    # Node sojourns — one per placed VNF.
    for vnf_id, node_id in plan.vnf_placements.items():
        node = g.nodes[node_id]
        cpu_capacity = float(node["cpu_capacity"])
        cpu_used = cpu_capacity - float(node["cpu_residual"])
        vnf = vnf_by_id[vnf_id]
        sojourn = node_sojourn(
            base_processing_delay=float(node["processing_delay"]),
            intensity=vnf.computational_intensity,
            cpu_capacity=cpu_capacity,
            cpu_used=cpu_used,
        )
        if math.isinf(sojourn):
            return math.inf
        total += sojourn

    # Link sojourns — one per link on each routed flow path.
    for flow in slice_req.flow_edges:
        path = plan.flow_routes.get((flow.source_vnf, flow.target_vnf), [])
        for link_id in path:
            u, v = _link_endpoints(g, link_id)
            edge = g[u][v]
            bw_capacity = float(edge["bandwidth_capacity"])
            bw_used = bw_capacity - float(edge["bw_residual"])
            sojourn = link_sojourn(
                propagation_delay=float(edge["propagation_delay"]),
                bandwidth_capacity=bw_capacity,
                bandwidth_used=bw_used,
            )
            if math.isinf(sojourn):
                return math.inf
            total += sojourn

    return total


def _check_c5b(
    slice_req: SliceRequest, plan: PlacementPlan
) -> tuple[float, bool]:
    """C5b — throughput floor check.

    Under v4 Eq. 3 the slice's bandwidth is fully derived from β_in:
    β_{k,k+1} = β_in · ∏_{j=1}^{k+1} ρ_{f_j}, and there is no independent
    throughput requirement. The slice meets its β_in iff every flow's
    per-link allocated BW ≥ its derived β_{k,k+1}. Any under-allocation
    means the slice did not receive the ingress rate it was contracted for.

    Returns:
        (effective_beta_in, violated)
            effective_beta_in: β_in scaled by the worst per-flow allocation
                ratio — useful for logging "how close was the slice".
            violated: True iff any flow was under-allocated.
    """
    target = slice_req.qos.min_throughput  # = β_in (set by slice_generator)
    if not slice_req.flow_edges:
        return target, False  # single-VNF slice: no flow-level BW constraint.

    worst_ratio = math.inf
    for flow in slice_req.flow_edges:
        per_link = plan.bw_allocations.get((flow.source_vnf, flow.target_vnf), {})
        if not per_link:
            # Same-node placement: no link allocation needed, contracts met.
            continue
        demand = flow.bandwidth_demand
        if demand <= 0:
            continue
        for bw in per_link.values():
            ratio = bw / demand
            worst_ratio = min(worst_ratio, ratio)

    if math.isinf(worst_ratio):
        # All flows were same-node — contracts met by construction.
        return target, False

    effective_beta_in = worst_ratio * target
    return effective_beta_in, worst_ratio < 1.0


def _count_inter_domain_hops(
    substrate: SubstrateNetwork,
    plan: PlacementPlan,
) -> int:
    """C9 — count inter-domain link traversals across all flow routes.

    Each distinct inter-domain link traversal contributes 1. If a flow
    crosses three inter-domain links, that's 3 hops.
    """
    g = substrate.graph
    hops = 0
    for path in plan.flow_routes.values():
        for link_id in path:
            u, v = _link_endpoints(g, link_id)
            link_type = g[u][v]["link_type"]
            if link_type == LinkType.INTER.value or link_type == LinkType.INTER:
                hops += 1
    return hops


def _link_endpoints(graph, link_id: str) -> tuple[str, str]:
    """Resolve a link_id to its (source, target) endpoints. Linear scan."""
    for u, v, d in graph.edges(data=True):
        if d["link_id"] == link_id:
            return u, v
    raise KeyError(f"link_id {link_id!r} not found in substrate")
