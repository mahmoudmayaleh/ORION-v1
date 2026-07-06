"""MDO pre-commit verification: E2E, BW, C5b, C7, C9, Cost(π*).

This is the MDO-side check using domain-reported values under the light-load
assumption. Distinct from the simulator's ground-truth post-commit check
that fires the hard penalty in the reward.
"""

from __future__ import annotations

from orion.actors.types import DomainResponse
from orion.mdo.types import InterDomainLink, ViolationInfo
from orion.types import QoSRequirements


def _pair_key(a: int, b: int) -> tuple[int, int]:
    """Normalise a domain pair to an undirected key (min, max).

    Inter-domain bandwidth is an undirected aggregate resource at the MDO's
    summary granularity — a link between domains a and b pools all physical
    edges regardless of the orientation they happen to be listed in. Keying
    by (min, max) makes the demand and residual maps line up by pair.
    """
    return (a, b) if a <= b else (b, a)


def inter_domain_demand_by_pair(
    partition: list[int],
    bw_demands: list[float],
) -> dict[tuple[int, int], float]:
    """Sum VCR-scaled β over cross-domain consecutive flows, grouped by pair.

    Two distinct chain edges that traverse the same domain-pair are summed
    into one demand, so the C5b check sees their joint load on the shared
    inter-domain link (the unsound per-edge alternative would let each pass
    individually while jointly overflowing).

    Args:
        partition: Per-VNF domain assignment.
        bw_demands: Per-flow-edge bandwidth demands (length = len(partition) - 1).

    Returns:
        Map from undirected domain-pair to total demanded bandwidth.
    """
    demand: dict[tuple[int, int], float] = {}
    for k in range(len(partition) - 1):
        a, b = partition[k], partition[k + 1]
        if a == b:
            continue
        beta = bw_demands[k] if k < len(bw_demands) else 0.0
        key = _pair_key(a, b)
        demand[key] = demand.get(key, 0.0) + beta
    return demand


def inter_domain_residual_by_pair(
    inter_domain_links: list[InterDomainLink],
) -> dict[tuple[int, int], float]:
    """Aggregate inter-domain link residuals into per-pair totals.

    Mirrors the aggregate granularity the MDO already works at: the policy
    observation and the C7 inter-domain delay are both per-domain-pair, so
    C5b is too. Exact per-link inter-domain bandwidth is the simulator's
    ground-truth job, not the MDO's.

    Args:
        inter_domain_links: Aggregated inter-domain link summaries (each may
            be listed in either orientation for a given pair).

    Returns:
        Map from undirected domain-pair to summed residual bandwidth.
    """
    residuals: dict[tuple[int, int], float] = {}
    for link in inter_domain_links:
        key = _pair_key(link.source_domain, link.target_domain)
        residuals[key] = residuals.get(key, 0.0) + link.bw_residual
    return residuals


def check_c5b_inter(
    demand_by_pair: dict[tuple[int, int], float],
    residual_by_pair: dict[tuple[int, int], float],
) -> bool:
    """Aggregate-sufficiency C5b check at inter-domain granularity.

    Returns True (violated) if any domain-pair's summed demand exceeds its
    summed residual. A pair with demand but no residual entry is treated as
    a violation (no inter-domain capacity exists for that pair).
    """
    for pair, demand in demand_by_pair.items():
        if demand > residual_by_pair.get(pair, 0.0):
            return True
    return False


def compute_e2e_delay(
    domain_responses: dict[int, DomainResponse],
    inter_domain_delays: dict[tuple[int, int], float],
    domain_sequence: list[int],
) -> float:
    """Compute E2E delay from domain actor responses (v6.2 Eq. 3).

    E2E = Σ_m δ^m + Σ_{ℓ ∈ L^inter} D_ℓ

    Args:
        domain_responses: Per-domain responses with intra_delay.
        inter_domain_delays: Propagation delay for inter-domain links,
            keyed by (src_domain, dst_domain).
        domain_sequence: Ordered list of domains traversed by the SFC.
            Derived from the partition: the sequence of unique domains
            in VNF placement order.

    Returns:
        Total E2E delay in ms.
    """
    # Sum intra-domain delays from each involved domain
    involved_domains = set(domain_sequence)
    total_intra = sum(
        domain_responses[d].intra_delay
        for d in involved_domains
        if d in domain_responses
    )

    # Sum inter-domain link delays along the domain sequence
    total_inter = 0.0
    for i in range(len(domain_sequence) - 1):
        src, dst = domain_sequence[i], domain_sequence[i + 1]
        if src != dst:
            total_inter += inter_domain_delays.get((src, dst), 0.0)

    return total_intra + total_inter


def compute_total_cost(
    domain_responses: dict[int, DomainResponse],
    inter_domain_bw: float,
    gamma_inter: float = 1.0,
) -> float:
    """Aggregate Cost(π*) from per-domain costs + inter-domain BW.

    Cost(π*) = Σ_m cost^m + γ_inter · Σ inter-domain BW

    Args:
        domain_responses: Per-domain responses with resource_cost.
        inter_domain_bw: Total bandwidth consumed on inter-domain links.
        gamma_inter: Weight for inter-domain bandwidth cost.

    Returns:
        Total placement cost.
    """
    intra_cost = sum(r.resource_cost for r in domain_responses.values())
    return intra_cost + gamma_inter * inter_domain_bw


def compute_inter_domain_bw(
    partition: list[int],
    bw_demands: list[float],
) -> float:
    """Compute total bandwidth on inter-domain links from the partition.

    For each flow edge (k, k+1), if partition[k] != partition[k+1],
    the flow crosses an inter-domain link with bandwidth bw_demands[k].

    Args:
        partition: Per-VNF domain assignment.
        bw_demands: Per-flow-edge bandwidth demands (length = len(partition) - 1).

    Returns:
        Total inter-domain bandwidth.
    """
    total = 0.0
    for k in range(len(partition) - 1):
        if partition[k] != partition[k + 1]:
            total += bw_demands[k] if k < len(bw_demands) else 0.0
    return total


def count_inter_domain_hops(partition: list[int]) -> int:
    """Count the number of inter-domain hops in the partition.

    A hop occurs when consecutive VNFs are placed in different domains.

    Args:
        partition: Per-VNF domain assignment.

    Returns:
        Number of inter-domain hops.
    """
    hops = 0
    for k in range(len(partition) - 1):
        if partition[k] != partition[k + 1]:
            hops += 1
    return hops


def domain_sequence_from_partition(partition: list[int]) -> list[int]:
    """Extract the ordered sequence of unique domains traversed by the SFC.

    Example: partition [0, 0, 1, 1, 2] -> domain_sequence [0, 1, 2]
    Example: partition [0, 1, 0, 1] -> domain_sequence [0, 1, 0, 1]

    Args:
        partition: Per-VNF domain assignment.

    Returns:
        Ordered list of domains, collapsing consecutive duplicates.
    """
    if not partition:
        return []
    seq = [partition[0]]
    for d in partition[1:]:
        if d != seq[-1]:
            seq.append(d)
    return seq


def precommit_check(  # DEAD PATH — coordinator routes cross-domain flows on physical edges now.

    partition: list[int],
    domain_responses: dict[int, DomainResponse],
    inter_domain_delays: dict[tuple[int, int], float],
    qos: QoSRequirements,
    bw_demands: list[float],
    max_inter_domain_hops: int = 3,
    gamma_inter: float = 1.0,
    inter_domain_residuals: dict[tuple[int, int], float] | None = None,
) -> tuple[bool, ViolationInfo, float, float]:
    """Run all pre-commit checks and return structured results.

    Checks:
        - All domain actors feasible (z^m = 1)
        - C7: E2E ≤ D^max_s
        - C5b: throughput floor (VCR-scaled minimum BW) — inter-domain part
        - C9: inter-domain hops ≤ max

    C5/C5b split (v6.2 constraint table). Intra-domain throughput is the
    domain actors' job: z^m = 1 already means every intra-domain flow routed
    with per-link BW reserved (see DomainActor.act — route_flow + allocate).
    The MDO owns only the INTER-domain part of C5b: that each domain-pair on
    the chain has enough aggregate residual bandwidth for the cross-domain
    flows traversing it. This is an aggregate-sufficiency check at summary
    granularity — exact per-link inter-domain BW is the simulator's
    ground-truth job, same as intra-domain C5.

    The MDO can still COMMIT a partition the simulator later rejects for C5
    (per-link) — by design; the hard penalty in the reward fires for the
    constraints the MDO owns (inter-domain C5b and C7), not C5.

    SCAFFOLD (Part B, Phase 5) — the residual this check reads is only as
    good as the accounting behind it. Inter-domain bandwidth is currently
    *costed* (compute_inter_domain_bw) but never *reserved*: nothing
    decrements inter-domain residuals across arrivals, so without a live
    per-pair residual counter `inter_domain_residuals` reflects capacity, not
    true residual, and this check is near-vacuous. Part B lands the
    reserve/release path with the substrate snapshot/restore and feeds the
    same per-pair counter here AND into the observation builder. Until then:
    do NOT benchmark inter-domain-tight topologies — both this check and the
    policy's observation run on capacity-not-residual. See
    docs/c5b_inter_domain_changeset.md.

    Args:
        partition: Per-VNF domain assignment.
        domain_responses: Responses from domain actors.
        inter_domain_delays: Propagation delays for inter-domain links.
        qos: QoS requirements for the slice.
        bw_demands: Per-flow-edge bandwidth demands.
        max_inter_domain_hops: C9 hop limit.
        gamma_inter: Inter-domain BW cost weight.
        inter_domain_residuals: Per-domain-pair aggregate residual bandwidth
            (undirected (min, max) keys, e.g. from inter_domain_residual_by_pair).
            When None, the inter-domain C5b check is skipped (no residual
            source available) and c5b_violated stays False.

    Returns:
        (passes, violation_info, e2e_delay, total_cost)
    """
    raise RuntimeError(
        "precommit_check is a dead path — the coordinator routes cross-domain "
        "flows on physical edges now. If you're seeing this, someone re-wired "
        "the aggregate per-pair scaffold back into the live path. Don't."
    )
    # Check all actors feasible
    actor_infeasible = any(
        not r.feasible for r in domain_responses.values()
    )

    # Domain sequence and E2E
    dom_seq = domain_sequence_from_partition(partition)
    e2e = compute_e2e_delay(domain_responses, inter_domain_delays, dom_seq)

    # Inter-domain metrics
    inter_bw = compute_inter_domain_bw(partition, bw_demands)
    inter_hops = count_inter_domain_hops(partition)

    # Inter-domain C5b: aggregate per-pair demand vs residual.
    if inter_domain_residuals is not None:
        demand_by_pair = inter_domain_demand_by_pair(partition, bw_demands)
        c5b_violated = check_c5b_inter(demand_by_pair, inter_domain_residuals)
    else:
        c5b_violated = False

    # Total cost
    total_cost = compute_total_cost(domain_responses, inter_bw, gamma_inter)

    # Build violation info
    violation = ViolationInfo(
        c5b_violated=c5b_violated,
        c7_violated=e2e > qos.max_e2e_delay,
        c9_violated=inter_hops > max_inter_domain_hops,
        actor_infeasible=actor_infeasible,
        e2e_delay=e2e,
        e2e_budget=qos.max_e2e_delay,
        total_bw=inter_bw,
        min_bw=qos.min_throughput,
        inter_domain_hops=inter_hops,
        max_inter_domain_hops=max_inter_domain_hops,
    )

    passes = not violation.has_violation
    return passes, violation, e2e, total_cost
