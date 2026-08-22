"""Encode the centralised critic's input s_t.

Choice A1: flat, concatenated tensor. Domains are consumed in the SAME
canonical order the MDO uses, `(tier_type, domain_id)`, so the flat
concatenation is permutation-invariant by construction without an
attention encoder.

Layout (single float vector):
    [ per-domain block (one per domain, canonical order):
        cpu_residual_frac, ram_residual_frac, cpu_capacity_norm,
        ram_capacity_norm, tier_dominant_idx_norm,
        bw_residual_frac_on_intra, active_slice_count_norm
    ] +
    [ per-inter-link block (one per inter-domain link, sorted by (src, dst)):
        bw_residual_frac, bw_capacity_norm, prop_delay_norm
    ] +
    [ global:
        total_arrivals_norm, admitted_count_norm, rejected_by_mdo_count_norm
    ] +
    [ arriving request (2026-08-20, REQUEST_FEAT_DIM slots):
        delay_budget_lin, delay_budget_log, min_throughput_norm,
        chain_len_norm, chain_cpu_norm, chain_ram_norm,
        max_vnf_cpu_norm, max_vnf_ram_norm
    ]

The shape depends on the substrate (num_domains, num_inter_domain_links),
so the trainer must probe once at start-up to determine the size, then
construct the critic with the right input dimension. This is the same
probe-and-bind pattern as `OrionSliceEnv.bind_observation_space`.

NOTE: this is `s_t` for V_φ(s_t), distinct from `o^MDO_t`. The MDO already
has its own canonical-order observation in `mdo/observation.py`; we reuse
that structure here for domain summaries (so the actor and critic see the
same ordering — non-negotiable per Choice A1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from orion.config import MDO_DELAY_REF
from orion.mdo.observation import build_domain_summaries, build_inter_domain_links
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier, TIER_INDEX, TIER_ORDER

# Same tier order as the MDO observation builder — must match exactly so
# the actor and critic see the same canonical layout (Choice A1).
# Canonical ordering lives in orion.types (one definition, see TIER_ORDER).
_TIER_ORDER = TIER_INDEX
_TIER_NORM = 3.0  # 4 tiers indexed 0..3 → normalize to [0, 1]


@dataclass
class GlobalStateStats:
    """Running normalisers for the per-episode arrival counts.

    The critic should see *normalised* counts so the input has bounded
    magnitude. These come from the runner's `EpisodeStats`.
    """

    total_arrivals: int = 0
    admitted: int = 0
    rejected_by_mdo: int = 0
    max_arrivals: int = 100  # one episode's expected horizon


#: Width of the arriving-request block appended to s_t (2026-08-20).
REQUEST_FEAT_DIM = 8


def _request_features(
    slice_req,
    max_cpu_cap: float,
    max_ram_cap: float,
    max_bw_cap: float,
) -> list[float]:
    """What is ARRIVING. Eight slots, all zero when no request is supplied.

    Why this exists (RL_DIAGNOSIS §6.2). s_t used to be substrate state plus
    three episode counters and nothing else, so V(s_t) was necessarily an
    average over the arrival distribution. Measured on 1885 L3 arrivals, a
    linear read of the whole 86-d s_t explained **R^2 = 0.038** of the actual
    RL reward, while the slice features it structurally excluded explained
    **0.215** -- the delay budget alone explained 0.103, nearly three times the
    entire critic input. Since A_a = r_a + gamma*V_{a+1} - V_a and V could not
    represent "a hard slice arrived", the overwhelming majority of every
    advantage was variance the action had no bearing on. Whitening removes the
    mean of that, not the variance.

    Same normalisers as the corresponding MDO observation block, so the actor's
    and the critic's views of the request agree (Choice A1 discipline).
    """
    if slice_req is None:
        return [0.0] * REQUEST_FEAT_DIM
    vnfs = getattr(slice_req, "vnfs", []) or []
    cpu = [float(v.cpu_demand) for v in vnfs]
    ram = [float(v.ram_demand) for v in vnfs]
    qos = getattr(slice_req, "qos", None)
    delay = float(getattr(qos, "max_e2e_delay", 0.0) or 0.0)
    thr = float(getattr(qos, "min_throughput", 0.0) or 0.0)
    return [
        min(1.0, delay / MDO_DELAY_REF) if delay > 0 else 0.0,
        (math.log1p(delay) / math.log1p(MDO_DELAY_REF)) if delay > 0 else 0.0,
        thr / max_bw_cap,
        len(vnfs) / 10.0,
        sum(cpu) / max_cpu_cap,
        sum(ram) / max_ram_cap,
        (max(cpu) / max_cpu_cap) if cpu else 0.0,
        (max(ram) / max_ram_cap) if ram else 0.0,
    ]


def encode_global_state(
    substrate: SubstrateNetwork,
    stats: GlobalStateStats,
    slice_req=None,
) -> torch.Tensor:
    """Encode s_t = (per-domain state, inter-domain link state, global state).

    The vector ordering is fixed at substrate construction and stable
    across the episode — exactly the same canonical-order invariant the
    MDO observation relies on.
    """
    parts: list[float] = []

    # Per-domain block — canonical order matches MDO observation.
    domains = build_domain_summaries(substrate)
    if domains:
        max_cpu_cap = max(s.cpu_capacity for s in domains) or 1.0
        max_ram_cap = max(s.ram_capacity for s in domains) or 1.0
    else:
        max_cpu_cap = max_ram_cap = 1.0

    for s in domains:
        cpu_frac = s.cpu_residual / s.cpu_capacity if s.cpu_capacity > 0 else 0.0
        ram_frac = s.ram_residual / s.ram_capacity if s.ram_capacity > 0 else 0.0

        parts.extend([
            cpu_frac,
            ram_frac,
            s.cpu_capacity / max_cpu_cap,
            s.ram_capacity / max_ram_cap,
            _TIER_ORDER.get(s.dominant_tier, 0) / _TIER_NORM,
            _intra_bw_residual_frac(substrate, s.domain_id),
            _active_slice_count_in_domain(substrate, s.domain_id) / 32.0,
        ])

    # Inter-domain link block.
    inter_links = build_inter_domain_links(substrate)
    if inter_links:
        max_bw_cap = max(l.bw_capacity for l in inter_links) or 1.0
        max_delay = max(l.propagation_delay for l in inter_links) or 1.0
    else:
        max_bw_cap = max_delay = 1.0

    for l in inter_links:
        bw_frac = l.bw_residual / l.bw_capacity if l.bw_capacity > 0 else 0.0
        parts.extend([
            bw_frac,
            l.bw_capacity / max_bw_cap,
            l.propagation_delay / max_delay,
        ])

    # Global block.
    norm = max(stats.max_arrivals, 1)
    parts.extend([
        stats.total_arrivals / norm,
        stats.admitted / norm,
        stats.rejected_by_mdo / norm,
    ])

    # Arriving-request block (2026-08-20). See `_request_features`.
    parts.extend(_request_features(slice_req, max_cpu_cap, max_ram_cap, max_bw_cap))

    return torch.tensor(parts, dtype=torch.float32)


def probe_global_state_dim(substrate: SubstrateNetwork) -> int:
    """Run the encoder once to determine the critic input dimension.

    Trainer calls this at episode start to size the CentralisedCritic.
    """
    return encode_global_state(substrate, GlobalStateStats()).numel()


# ── Internals ────────────────────────────────────────────────────────────


def _intra_bw_residual_frac(substrate: SubstrateNetwork, domain_id: int) -> float:
    """Mean residual BW fraction across intra-domain links of `domain_id`."""
    g = substrate.graph
    nodes = set(substrate.nodes_in_domain(domain_id))
    if not nodes:
        return 0.0
    total = 0.0
    cap = 0.0
    for u, v, d in g.edges(data=True):
        if u in nodes and v in nodes:
            total += d["bw_residual"]
            cap += d["bandwidth_capacity"]
    return float(total / cap) if cap > 0 else 0.0


def _active_slice_count_in_domain(substrate: SubstrateNetwork, domain_id: int) -> int:
    """Number of active slices with ≥1 VNF placed in this domain."""
    count = 0
    for plan, _ in substrate._active_slices.values():
        if any(
            substrate.graph.nodes[node]["domain_id"] == domain_id
            for node in plan.vnf_placements.values()
        ):
            count += 1
    return count
