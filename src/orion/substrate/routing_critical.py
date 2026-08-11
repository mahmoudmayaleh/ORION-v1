"""Routing-critical topology family `C+_T-_B-_RC` (§Q, pre-registered 2026-07-15).

Concentrates the two admittable-but-missed failure classes from the diagnostic
taxonomy (PREREG_AMENDMENT_2026-07-15_Q.md §Q.4):

  1. Inter-domain bandwidth traps — a feasible placement exists only via a
     non-obvious cut point that avoids saturating a bottleneck inter-domain
     link; the greedy/naive cut saturates it and rejects.
  2. Cut-point-sensitive chains — admission depends on WHICH VNF boundary
     crosses a domain, because VCR scaling ramps the per-flow bandwidth demand
     ×1 → ×3 along the chain, so a late cut costs ~3× the inter-domain BW of an
     early one.

Committed parameters (Q.4 — one draw, gen_seed=20260715):
  num_domains=5, nodes_per_domain=4, tier HOSTILE (edge/cloud split), capacity
  friendly (CPU not the binding constraint), inter_domain_links=2, inter-domain
  BW range (60,120) with a per-instance override in {70,90,110}, chain length
  4–6, VCR ramp ×1→×3 (per-VNF vcr≈1.25), tiers spanning the split so the cut
  point is free.

The substrate forces ≥1 cut (a 5–6-VNF chain's CPU exceeds one 4-node domain)
but leaves the cut position free (VNFs permitted on both mec and regional_cloud,
which live in odd and even domains respectively). The VCR ramp then makes late
cuts saturate the tight inter-domain link — that is the trap.
"""

from __future__ import annotations

import numpy as np

from orion.config import TopologyConfig
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.substrate.topology_families import (
    _enforce_tier_split,
    compute_signature,
    TopologySignature,
)
from orion.sim.slice_generator import _resolve_permitted_nodes
from orion.types import (
    VNF,
    FlowEdge,
    SliceRequest,
    QoSRequirements,
    SliceType,
)

RC_FAMILY_SHORT = "C+_T-_B-_RC"
RC_SPEC_VERSION = "v2"    # RC-v2, Q.4 Δ (2026-07-15). RC-v1 (gen_seed 20260715,
                          # ramp ×3, all-forced) was degenerate (Plain=0) — kept
                          # in the record as the boundary draw, not run.
RC_GEN_SEED = 20260716    # v2 fresh draw (was 20260715 for v1)
RC_BW_OVERRIDES = (70.0, 90.0, 110.0)  # swept across seeds 42/43/44 (Q.4, unchanged)

# ── Committed chain parameters (Q.4, amended RC-v2) ─────────────────────────
_RC_CHAIN_MIN = 4
_RC_CHAIN_MAX = 6
# RC-v2 change 1: VCR ramp ×1→×2 (was ×1.25→×3). 2^(1/5)=1.1487 → over 3–5 edges
# the tail lands ~β×1.5–2.0, INSIDE the intra bw_range (60–120), so co-location
# of adjacent tail VNFs is sometimes viable and Plain produces placements.
_RC_VCR = 1.15
_RC_CPU_RANGE = (3.0, 4.0)
_RC_RAM_RANGE = (4.0, 8.0)
_RC_INTENSITY = 0.7       # moderate → C7 delay is not the binding constraint
_RC_BETA_IN = (40.0, 55.0)  # head ~β×1.15; tail ~β×1.5–2.0 → straddles the caps
_RC_DELAY = (30.0, 80.0)  # generous → routing/BW binds, not delay
# RC-v2 change 2: mixed forcing — the family CONTAINS the trap, not consists of
# it. ~60% of arrivals are edge-head/cloud-tail forced (trap lives); ~40% are
# tier-flexible (co-location viable). Also the honest model of a routing-critical
# regime: hard arrivals arrive among ordinary ones.
_RC_FORCE_FRACTION = 0.60
# Tier assignment forces a cross-domain cut while leaving the cut POSITION free.
# After the hostile split, odd domains hold edge tiers, even domains hold cloud
# tiers. An edge-only head and a cloud-only tail force the chain to span an odd
# and an even domain (≥1 inter-domain crossing); the flexible middle lets the
# single transition sit at any boundary. The VCR ramp then makes late crossings
# (bw ≈ ×3) saturate the tight inter-domain link while an early crossing fits.
_RC_TIERS_EDGE = ["ran_edge", "mec"]           # odd domains only
_RC_TIERS_FLEX = ["mec", "regional_cloud"]      # either side — free cut
_RC_TIERS_CLOUD = ["regional_cloud", "central_cloud"]  # even domains only


def _rc_tiers_for_position(k: int, n: int) -> list[str]:
    if k == 0:
        return _RC_TIERS_EDGE
    if k == n - 1:
        return _RC_TIERS_CLOUD
    return _RC_TIERS_FLEX


def rc_topology_config(num_domains: int = 5) -> TopologyConfig:
    """The RC-v2 TopologyConfig (Q.4 committed params). Exposed so BC actor
    pretraining (§R) can build its dataset on the same topology."""
    return TopologyConfig(
        num_domains=num_domains,
        nodes_per_domain=[4] * num_domains,          # moderate capacity
        intra_link_density=0.5,
        inter_domain_links=2,                         # few bottleneck links
        tier_distribution={                           # hostile split seed
            "ran_edge": 0.10, "mec": 0.10,
            "regional_cloud": 0.40, "central_cloud": 0.40,
        },
        bw_range=(60.0, 120.0),                       # tight inter-domain BW
        delay_intra_range=(0.5, 5.0),
        delay_inter_range=(5.0, 20.0),
    )


def generate_rc_instance(
    seed: int,
    inter_domain_bw_override: float | None = None,
    num_domains: int = 5,
) -> SubstrateNetwork:
    """Generate one routing-critical substrate instance (Q.4 committed params)."""
    rng = np.random.default_rng(seed)
    config = rc_topology_config(num_domains)
    sub = generate_multi_domain_topology(config, rng)
    _enforce_tier_split(sub, rng)                     # edge tiers odd, cloud even

    if inter_domain_bw_override is not None:
        g = sub.graph
        for u, v, d in g.edges(data=True):
            u_dom = g.nodes[u].get("domain_id", -1)
            v_dom = g.nodes[v].get("domain_id", -1)
            if u_dom != v_dom and u_dom >= 0 and v_dom >= 0:
                d["bandwidth_capacity"] = inter_domain_bw_override
                d["bw_capacity"] = inter_domain_bw_override
                d["bw_residual"] = inter_domain_bw_override
    return sub


def rc_signature(sub: SubstrateNetwork) -> TopologySignature:
    return compute_signature(sub, family_name=RC_FAMILY_SHORT)


def rc_slice_factory(
    request_id: str,
    substrate: SubstrateNetwork,
    rng: np.random.Generator,
    arrival_time: float = 0.0,
    lifetime: float = 0.0,
) -> SliceRequest:
    """Cut-sensitive chain with a ×1→×3 VCR bandwidth ramp (Q.4).

    Same call signature as `generate_slice_request`, so it drops straight into
    `ArrivalProcess(slice_factory=...)` and is seen identically by the ceiling
    enumerator and every approach.

    Carried under `SliceType.EMBB` (the slice_type is only a cache-key coarse
    bucket); the RC chains are distinguished by their `sfc_template` — the
    ordered `vnf_type` tuple `("RCF0","RCF1",...)` — so a given chain length is
    one stable plan_signature (cacheable).
    """
    # RC-v2: per-arrival coin — forced (edge-head/cloud-tail, the trap) vs
    # tier-flexible (co-location viable). ~60% forced.
    forced = bool(rng.random() < _RC_FORCE_FRACTION)
    n_vnfs = int(rng.integers(_RC_CHAIN_MIN, _RC_CHAIN_MAX + 1))

    vnfs: list[VNF] = []
    for k in range(n_vnfs):
        cpu = float(rng.uniform(*_RC_CPU_RANGE))
        ram = float(rng.uniform(*_RC_RAM_RANGE))
        tiers = _rc_tiers_for_position(k, n_vnfs) if forced else _RC_TIERS_FLEX
        permitted = _resolve_permitted_nodes(tiers, substrate)
        vnfs.append(VNF(
            vnf_id=f"{request_id}_f{k}",
            vnf_type=f"RCF{k}",
            cpu_demand=round(cpu, 1),
            ram_demand=round(ram, 1),
            permitted_nodes=permitted,
            computational_intensity=_RC_INTENSITY,
            vcr=_RC_VCR,
        ))

    beta_in = float(rng.uniform(*_RC_BETA_IN))
    flow_edges: list[FlowEdge] = []
    for k in range(len(vnfs) - 1):
        # β_{k,k+1} = β_in · ∏_{j=1}^{k+1} ρ_{f_j}  (v4 Eq. 3); ρ=1.25 ⇒ ramp.
        vcr_product = 1.0
        for j in range(k + 1):
            vcr_product *= vnfs[j].vcr
        flow_edges.append(FlowEdge(
            source_vnf=vnfs[k].vnf_id,
            target_vnf=vnfs[k + 1].vnf_id,
            bandwidth_demand=round(beta_in * vcr_product, 1),
        ))

    delay = float(rng.uniform(*_RC_DELAY))
    return SliceRequest(
        request_id=request_id,
        slice_type=SliceType.EMBB,
        vnfs=vnfs,
        flow_edges=flow_edges,
        qos=QoSRequirements(
            max_e2e_delay=round(delay, 1),
            min_throughput=round(beta_in, 1),
        ),
        arrival_time=arrival_time,
        lifetime=lifetime,
    )
