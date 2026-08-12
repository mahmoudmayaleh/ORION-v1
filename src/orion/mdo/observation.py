"""Build the MDO observation from substrate state and the abstract plan.

The observation follows v6.2 Eq. 2:
  o^MDO_t = ({ĉ^m_res, r̂^m_res, τ^m}, {b^res_ℓ, D_ℓ}, π̃_t)

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
)
from orion.config import MDO_HEADROOM_CPU_REF, MDO_HEADROOM_RAM_REF
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import (InfrastructureTier, TIER_INDEX, TIER_INDEX_NORM,
                         TIER_ORDER)

# Canonical tier ordering for sort key
# Canonical ordering lives in orion.types (one definition, see TIER_ORDER).
_TIER_ORDER = TIER_INDEX

# Feature dimensions. THIS module owns the layout; `hierarchical_topology.OBS_DIM`
# imports DOMAIN_FEAT_DIM from here rather than restating it, so the declared width
# and the emitted width cannot drift apart.
#
# Per domain: cpu_res_frac, ram_res_frac, cpu_cap_norm, ram_cap_norm,
# tier_dominant_idx, then PER TIER: residual CPU, residual RAM (§Y.1e, 2026-07-31)
# and the best-fitting node's residual CPU and RAM (h^m, restored 2026-08-12).
#
# On h^m. It was removed on 2026-08-11 on the argument that a node-level statistic
# does not belong in a domain summary. That argument was about where the number comes
# from, not about what a domain may publish: an operator advertising "I can host one
# VNF of up to this size" is publishing a capability, not a layout, and every practical
# federation interface carries some form of it. Removing it also made the comparison
# unequal in a way that mattered more than the principle, since the full-observability
# baselines read node residuals directly and could always answer the question h^m
# answers. It is restored, and it is now published on the SAME surface to all three
# consumers of the abstract view: this tensor, the LLM planner
# (llm/abstract_topology.build_abstract_topology) and the partial-observability
# heuristic (scripts/partial_obs_prior).
#
# Why per-tier. An aggregate residual cannot express "this domain's edge tier is
# full but its regional tier is not", which under heterogeneous domain composition
# is the common case. Measured cost of not having it: `actor_infeasible` rejections,
# where the orchestrator picks a domain that looks adequate in aggregate and the
# domain actor then cannot place the chain on real nodes, ran 59 / 250 / 494 / 711
# out of 2000 arrivals at L1-L4 for the partial-observability heuristic, against
# ZERO for the same heuristic reading node residuals.
#
# Scope is deliberately narrow. This is INFORMATION ONLY: the coordinator stays
# single-attempt, one partition decision per arrival, no retry and no per-VNF
# reassignment. The orchestrator gets what it needs to avoid a bad partition, not a
# mechanism to recover from one.
#
# No presence flag and no per-tier capacity: an absent tier and an exhausted tier
# both read zero. That is correct because composition is FIXED, so an absent tier
# reads zero in every instance and every episode and is learnable from the domain's
# identity. A flag would cost width and buy nothing.
DOMAIN_FEAT_DIM = 5 + 4 * len(TIER_ORDER)   # 17
LINK_FEAT_DIM = 3     # bw_res_frac, bw_cap_norm, delay_norm
VNF_FEAT_DIM = 5      # cpu_norm, ram_norm, tier_idx, vcr, bw_norm

# Per VNF slot the observation ALSO carries a one-hot of the proposed domain m̃_k, in
# CANONICAL index space, so the plan block is VNF_FEAT_DIM + M wide (2026-08-12).
#
# Why the proposal is state. Until now m̃ reached the policy only as an additive bonus
# on its own output logits at decision time (`prior_weight` in AutoregMDOPolicy.forward),
# and, during training, through a KL term to the proposer that was measured inert
# (agreement 0.007 at beta=25; trained argmax worse than untrained). A bias applied
# after the fact cannot be learned from: the policy could not represent "a proposal was
# made, and it was this one", so it could never learn WHEN the proposal is worth
# following and when to override it. It could only converge to agreeing more often.
#
# As an input it can. The policy conditions on the proposal together with the substrate
# state, so "the proposer suggests co-locating here, but this tier's largest free node
# is too small" is expressible, and deviation becomes a learned decision rather than an
# absence of influence. This is what makes "the planner proposes, the policy disposes" a
# measurable relationship rather than a description.
#
# Every approach's observation carries the m̃ of ITS OWN plan source, so the width is
# constant and the arms differ in who proposes, not in whether anyone does. An
# all-zero block means no proposal was made for that slot (padding beyond the chain).
PLAN_ONEHOT = True


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

        # Per-tier residuals (§Y.1e). Every tier gets an entry, including tiers this
        # domain does not hold, which read 0.0: the observation is a fixed-width
        # vector, so a missing key and a zero must not be different shapes.
        tier_cpu = {t: 0.0 for t in TIER_ORDER}
        tier_ram = {t: 0.0 for t in TIER_ORDER}
        # h^m per tier: the best-fitting node, scored min(cpu/CPU_REF, ram/RAM_REF) so a
        # node rich in one resource and starved in the other cannot win. The winner's OWN
        # residuals are reported, so the pair always describes one real node.
        best_fit = {t: -1.0 for t in TIER_ORDER}
        tier_hcpu = {t: 0.0 for t in TIER_ORDER}
        tier_hram = {t: 0.0 for t in TIER_ORDER}
        for n in nodes:
            t = InfrastructureTier(g.nodes[n]["tier"])
            n_cpu = g.nodes[n]["cpu_residual"]
            n_ram = g.nodes[n]["ram_residual"]
            tier_cpu[t] += n_cpu
            tier_ram[t] += n_ram
            fit = min(n_cpu / MDO_HEADROOM_CPU_REF, n_ram / MDO_HEADROOM_RAM_REF)
            if fit > best_fit[t]:
                best_fit[t] = fit
                tier_hcpu[t] = n_cpu
                tier_hram[t] = n_ram

        summaries.append(DomainSummary(
            domain_id=domain_id,
            dominant_tier=InfrastructureTier(dominant),
            cpu_residual=cpu_res,
            ram_residual=ram_res,
            cpu_capacity=cpu_cap,
            ram_capacity=ram_cap,
            supported_tiers=[InfrastructureTier(t) for t in unique_tiers],
            tier_cpu_residual=tier_cpu,
            tier_ram_residual=tier_ram,
            tier_max_node_cpu=tier_hcpu,
            tier_max_node_ram=tier_hram,
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
) -> MDOObservation:
    """Build the structured MDO observation."""
    return MDOObservation(
        domain_summaries=build_domain_summaries(substrate),
        inter_domain_links=build_inter_domain_links(substrate),
        plan=plan,
    )


def observation_to_tensor(obs: MDOObservation, max_vnfs: int = 10) -> torch.Tensor:
    """Flatten the MDO observation into a fixed-size tensor for the policy.

    Layout: [domain_features | link_features | plan_features | reserved]

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
            _TIER_ORDER.get(s.dominant_tier, 0) / TIER_INDEX_NORM,
        ])
        # Per-tier residuals (§Y.1e), normalised by the SAME whole-substrate maxima
        # as the aggregates above so the two blocks are on one scale. Iterated over
        # TIER_ORDER, not over the dict, so the slot for a given tier is at a fixed
        # offset in every domain's block even when a domain lacks that tier.
        # h^m rides on the SAME two maxima, so "this tier holds 400 CPU in total" and
        # "its biggest free node holds 12 of them" are directly comparable numbers.
        for t in TIER_ORDER:
            parts.append(s.tier_cpu_residual.get(t, 0.0) / max_cpu_cap)
            parts.append(s.tier_ram_residual.get(t, 0.0) / max_ram_cap)
            parts.append(s.tier_max_node_cpu.get(t, 0.0) / max_cpu_cap)
            parts.append(s.tier_max_node_ram.get(t, 0.0) / max_ram_cap)

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

    # m̃ is emitted in CANONICAL index space, the same space the policy's logits and the
    # tier mask live in. suggested_domains holds raw domain ids; feeding those straight
    # in would misalign the one-hot with the action whenever the canonical sort differs
    # from the id order, which is exactly the §O.5 defect that voided the earlier
    # agreement numbers.
    domain_to_canonical = {s.domain_id: i for i, s in enumerate(obs.domain_summaries)}
    n_dom = len(obs.domain_summaries)
    suggested = list(getattr(obs.plan, "suggested_domains", None) or [])

    for k in range(max_vnfs):
        if k < obs.plan.num_vnfs:
            tier_idx = _TIER_ORDER.get(obs.plan.required_tiers[k], 0)
            parts.extend([
                obs.plan.cpu_demands[k] / max_cpu_d,
                obs.plan.ram_demands[k] / max_ram_d,
                tier_idx / TIER_INDEX_NORM,
                obs.plan.vcrs[k],
                obs.plan.bw_demands[k] / max_bw_d if k < len(obs.plan.bw_demands) else 0.0,
            ])
            onehot = [0.0] * n_dom
            if k < len(suggested):
                c = domain_to_canonical.get(suggested[k])
                if c is not None:
                    onehot[c] = 1.0
            parts.extend(onehot)
        else:
            parts.extend([0.0] * (VNF_FEAT_DIM + n_dom))

    # Reserved feature slots (constant zero). Kept so the observation width
    # stays stable across checkpointed policies.
    parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    return torch.tensor(parts, dtype=torch.float32)


# §U.1d (2026-07-18): the legacy mask below marks a domain feasible for VNF k iff
# a SINGLE derived required_tier (perm[0]) is in the domain — a lossy compression of
# the VNF's multi-tier permitted set D_f (permitted_nodes), which excluded ~37% of
# feasible domains in every RL run. Node-based feasibility (permitted_nodes ∩ domain
# nodes) is the truest, matching the actor and the C8 checker. Default OFF keeps every
# legacy path byte-identical; RC/§U.1 sets it True.
USE_NODE_BASED_TIER_MASK = False


def build_tier_masks(
    plan: PlanSummary,
    domain_summaries: list[DomainSummary],
    permitted_nodes: list | None = None,
    substrate: SubstrateNetwork | None = None,
) -> torch.Tensor:
    """Build boolean mask [K, M] of feasible domains per VNF (canonical order).

    True means VNF k can be placed in domain m. This is the ONLY hard mask at
    the MDO level. §U.1d: when `USE_NODE_BASED_TIER_MASK` and permitted_nodes +
    substrate are supplied, feasibility is NODE-based — domain m holds at least one
    of VNF k's permitted nodes (D_f ∩ nodes_in_domain(m) ≠ ∅). Otherwise falls back
    to the legacy single-required-tier check.
    """
    K = plan.num_vnfs
    M = len(domain_summaries)
    mask = torch.zeros(K, M, dtype=torch.bool)

    if USE_NODE_BASED_TIER_MASK and permitted_nodes is not None and substrate is not None:
        dom_nodes = [set(substrate.nodes_in_domain(s.domain_id)) for s in domain_summaries]
        for k in range(K):
            pk = set(permitted_nodes[k]) if k < len(permitted_nodes) else set()
            for m in range(M):
                if pk & dom_nodes[m]:
                    mask[k, m] = True
        return mask

    for k in range(K):
        required = plan.required_tiers[k]
        for m, summary in enumerate(domain_summaries):
            if required in summary.supported_tiers:
                mask[k, m] = True

    return mask
