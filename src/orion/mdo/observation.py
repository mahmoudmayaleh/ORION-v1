"""Build the MDO observation from substrate state, abstract plan, and retry history.

The observation follows v6.2 Eq. 2:
  o^MDO_t = ({ĉ^m_res, r̂^m_res, τ^m}, {b^res_ℓ, D_ℓ}, π̃_t, h_t)

Domain features are sorted by canonical key (tier_type, domain_id) for
stable ordering across state evolution. Do NOT sort by state-dependent
quantities (residual capacity etc).
"""

from __future__ import annotations

from collections import Counter

import torch

from orion.mdo.types import (
    DomainSummary,
    InterDomainLink,
    MDOObservation,
    PlanSummary,
    RetryHistory,
)
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier

# Canonical tier ordering for sort key
_TIER_ORDER = {
    InfrastructureTier.RAN_EDGE: 0,
    InfrastructureTier.MEC: 1,
    InfrastructureTier.REGIONAL_CLOUD: 2,
    InfrastructureTier.CENTRAL_CLOUD: 3,
}

# Feature dimensions
DOMAIN_FEAT_DIM = 5   # cpu_res_frac, ram_res_frac, cpu_cap_norm, ram_cap_norm, tier_dominant_idx
LINK_FEAT_DIM = 3     # bw_res_frac, bw_cap_norm, delay_norm
VNF_FEAT_DIM = 5      # cpu_norm, ram_norm, tier_idx, vcr, bw_norm


def _dominant_tier(tiers: list[str]) -> InfrastructureTier:
    """Return the most common tier in a domain, breaking ties by tier order."""
    counts = Counter(tiers)
    return max(
        counts.keys(),
        key=lambda t: (counts[t], -_TIER_ORDER.get(InfrastructureTier(t), 99)),
    )


def build_domain_summaries(substrate: SubstrateNetwork) -> list[DomainSummary]:
    """Build per-domain summaries, sorted by canonical key (tier_type, domain_id)."""
    summaries = []
    g = substrate.graph

    for domain_id in range(substrate.num_domains):
        nodes = substrate.nodes_in_domain(domain_id)
        if not nodes:
            continue

        cpu_res = sum(g.nodes[n]["cpu_residual"] for n in nodes)
        ram_res = sum(g.nodes[n]["ram_residual"] for n in nodes)
        cpu_cap = sum(g.nodes[n]["cpu_capacity"] for n in nodes)
        ram_cap = sum(g.nodes[n]["ram_capacity"] for n in nodes)

        tiers_in_domain = [g.nodes[n]["tier"] for n in nodes]
        unique_tiers = sorted(set(tiers_in_domain), key=lambda t: _TIER_ORDER.get(InfrastructureTier(t), 99))
        dominant = _dominant_tier(tiers_in_domain)

        summaries.append(DomainSummary(
            domain_id=domain_id,
            dominant_tier=InfrastructureTier(dominant),
            cpu_residual=cpu_res,
            ram_residual=ram_res,
            cpu_capacity=cpu_cap,
            ram_capacity=ram_cap,
            supported_tiers=[InfrastructureTier(t) for t in unique_tiers],
        ))

    # Sort by canonical key: (tier_type, domain_id) — stable across state evolution
    summaries.sort(key=lambda s: (_TIER_ORDER.get(s.dominant_tier, 99), s.domain_id))
    return summaries


def build_inter_domain_links(substrate: SubstrateNetwork) -> list[InterDomainLink]:
    """Build aggregated inter-domain link summaries."""
    g = substrate.graph
    link_agg: dict[tuple[int, int], dict] = {}

    for u, v, d in g.edges(data=True):
        src_dom = g.nodes[u]["domain_id"]
        dst_dom = g.nodes[v]["domain_id"]
        if src_dom == dst_dom:
            continue

        key = (src_dom, dst_dom)
        if key not in link_agg:
            link_agg[key] = {"bw_res": 0.0, "bw_cap": 0.0, "delay": float("inf")}

        link_agg[key]["bw_res"] += d["bw_residual"]
        link_agg[key]["bw_cap"] += d["bandwidth_capacity"]
        link_agg[key]["delay"] = min(link_agg[key]["delay"], d["propagation_delay"])

    links = []
    for (src, dst), agg in sorted(link_agg.items()):
        links.append(InterDomainLink(
            source_domain=src,
            target_domain=dst,
            bw_residual=agg["bw_res"],
            bw_capacity=agg["bw_cap"],
            propagation_delay=agg["delay"],
        ))
    return links


def build_mdo_observation(
    substrate: SubstrateNetwork,
    plan: PlanSummary,
    retry_history: RetryHistory | None = None,
) -> MDOObservation:
    """Build the structured MDO observation."""
    return MDOObservation(
        domain_summaries=build_domain_summaries(substrate),
        inter_domain_links=build_inter_domain_links(substrate),
        plan=plan,
        retry_history=retry_history,
    )


def observation_to_tensor(obs: MDOObservation, max_vnfs: int = 10) -> torch.Tensor:
    """Flatten the MDO observation into a fixed-size tensor for the policy.

    Layout: [domain_features | link_features | plan_features | retry_stats]

    Domain features are already canonically sorted. Plan features are
    padded to max_vnfs so the tensor has constant size regardless of
    the actual number of VNFs in the slice.
    """
    parts = []

    # Global normalization constants
    max_cpu_cap = max((s.cpu_capacity for s in obs.domain_summaries), default=1.0) or 1.0
    max_ram_cap = max((s.ram_capacity for s in obs.domain_summaries), default=1.0) or 1.0
    max_bw_cap = max((l.bw_capacity for l in obs.inter_domain_links), default=1.0) or 1.0
    max_delay = max((l.propagation_delay for l in obs.inter_domain_links), default=1.0) or 1.0

    # Domain features (already sorted by canonical key)
    for s in obs.domain_summaries:
        cpu_frac = s.cpu_residual / s.cpu_capacity if s.cpu_capacity > 0 else 0.0
        ram_frac = s.ram_residual / s.ram_capacity if s.ram_capacity > 0 else 0.0
        parts.extend([
            cpu_frac,
            ram_frac,
            s.cpu_capacity / max_cpu_cap,
            s.ram_capacity / max_ram_cap,
            _TIER_ORDER.get(s.dominant_tier, 0) / 3.0,  # normalized tier index
        ])

    # Inter-domain link features
    for l in obs.inter_domain_links:
        bw_frac = l.bw_residual / l.bw_capacity if l.bw_capacity > 0 else 0.0
        parts.extend([
            bw_frac,
            l.bw_capacity / max_bw_cap,
            l.propagation_delay / max_delay,
        ])

    # Plan features (per-VNF)
    max_cpu_d = max(obs.plan.cpu_demands, default=1.0) or 1.0
    max_ram_d = max(obs.plan.ram_demands, default=1.0) or 1.0
    max_bw_d = max(obs.plan.bw_demands, default=1.0) or 1.0

    for k in range(max_vnfs):
        if k < obs.plan.num_vnfs:
            tier_idx = _TIER_ORDER.get(obs.plan.required_tiers[k], 0)
            parts.extend([
                obs.plan.cpu_demands[k] / max_cpu_d,
                obs.plan.ram_demands[k] / max_ram_d,
                tier_idx / 3.0,
                obs.plan.vcrs[k],
                obs.plan.bw_demands[k] / max_bw_d if k < len(obs.plan.bw_demands) else 0.0,
            ])
        else:
            parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # Retry statistics: aggregate-only encoding (count + per-violation-type rate).
    # Design choice: for N_part = 3-5, aggregate statistics are sufficient.
    # A per-attempt flatten with padding or a set transformer over attempts
    # would be order-invariant but adds complexity for marginal benefit at
    # this scale. The policy conditions on previous failures through these
    # aggregate rates, not through an LSTM or recurrent module — this is
    # correct for the per-arrival one-shot decision framing (no recurrence
    # needed since h_t is fully captured in the observation).
    if obs.retry_history and obs.retry_history.num_attempts > 0:
        h = obs.retry_history
        parts.append(h.num_attempts / 10.0)  # normalized attempt count
        # Fraction of attempts with each violation type
        vecs = h.last_violation_vectors(h.num_attempts)
        if vecs:
            parts.append(sum(v[0] for v in vecs) / len(vecs))  # C5b rate
            parts.append(sum(v[1] for v in vecs) / len(vecs))  # C7 rate
            parts.append(sum(v[2] for v in vecs) / len(vecs))  # C9 rate
            parts.append(sum(v[3] for v in vecs) / len(vecs))  # actor infeasible rate
        else:
            parts.extend([0.0, 0.0, 0.0, 0.0])
    else:
        parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    return torch.tensor(parts, dtype=torch.float32)


def build_tier_masks(
    plan: PlanSummary,
    domain_summaries: list[DomainSummary],
) -> torch.Tensor:
    """Build boolean mask [K, M] of tier-feasible domains per VNF.

    True means VNF k can be placed in domain m (the domain supports the
    required tier). This is the ONLY hard mask at the MDO level.
    """
    K = plan.num_vnfs
    M = len(domain_summaries)
    mask = torch.zeros(K, M, dtype=torch.bool)

    for k in range(K):
        required = plan.required_tiers[k]
        for m, summary in enumerate(domain_summaries):
            if required in summary.supported_tiers:
                mask[k, m] = True

    return mask
