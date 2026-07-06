"""Slice request generator producing SFC-structured requests with QoS profiles.

Generates realistic slice requests for the 5 service categories (eMBB, URLLC,
mMTC, V2X, XR) with appropriate VNF chains, resource demands, and QoS
constraints. Adapted from Virne's VirtualNetworkRequestSimulator pattern but
structured as SFCs rather than arbitrary virtual network topologies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orion.substrate.graph_model import SubstrateNetwork
from orion.types import (
    VNF,
    FlowEdge,
    InfrastructureTier,
    QoSRequirements,
    SliceRequest,
    SliceType,
)

# ── Per-slice-type SFC templates ──────────────────────────────────────────────

_VNF_TEMPLATES: dict[SliceType, list[dict]] = {
    SliceType.EMBB: [
        {"type": "Firewall",   "cpu": (2, 4),  "ram": (2, 8),  "intensity": 0.8, "vcr": 1.0, "tiers": ["mec", "regional_cloud", "central_cloud"]},
        {"type": "CDN",        "cpu": (4, 8),  "ram": (8, 16), "intensity": 1.2, "vcr": 0.7, "tiers": ["mec", "regional_cloud"]},
        {"type": "vEPC",       "cpu": (4, 8),  "ram": (4, 16), "intensity": 1.0, "vcr": 1.0, "tiers": ["regional_cloud", "central_cloud"]},
    ],
    SliceType.URLLC: [
        {"type": "Firewall",   "cpu": (1, 2),  "ram": (1, 4),  "intensity": 0.5, "vcr": 1.0, "tiers": ["ran_edge", "mec"]},
        {"type": "vUPF",       "cpu": (2, 4),  "ram": (2, 8),  "intensity": 0.6, "vcr": 1.0, "tiers": ["ran_edge", "mec"]},
    ],
    SliceType.MMTC: [
        {"type": "IoTGateway", "cpu": (1, 2),  "ram": (1, 4),  "intensity": 0.4, "vcr": 0.3, "tiers": ["ran_edge", "mec"]},
        {"type": "Aggregator", "cpu": (2, 4),  "ram": (2, 8),  "intensity": 0.6, "vcr": 0.5, "tiers": ["mec", "regional_cloud"]},
        {"type": "Analytics",  "cpu": (4, 8),  "ram": (8, 16), "intensity": 1.5, "vcr": 1.0, "tiers": ["regional_cloud", "central_cloud"]},
    ],
    SliceType.V2X: [
        {"type": "Firewall",   "cpu": (1, 2),  "ram": (1, 4),  "intensity": 0.5, "vcr": 1.0, "tiers": ["ran_edge", "mec"]},
        {"type": "V2XController", "cpu": (2, 4), "ram": (4, 8), "intensity": 0.7, "vcr": 1.0, "tiers": ["mec"]},
        {"type": "vEPC",       "cpu": (2, 4),  "ram": (2, 8),  "intensity": 1.0, "vcr": 1.0, "tiers": ["regional_cloud"]},
    ],
    SliceType.XR: [
        {"type": "Firewall",   "cpu": (2, 4),  "ram": (2, 8),  "intensity": 0.8, "vcr": 1.0, "tiers": ["mec"]},
        {"type": "MediaProc",  "cpu": (8, 16), "ram": (16, 32), "intensity": 2.0, "vcr": 1.2, "tiers": ["mec", "regional_cloud"]},
        {"type": "CDN",        "cpu": (4, 8),  "ram": (8, 16), "intensity": 1.2, "vcr": 0.7, "tiers": ["regional_cloud", "central_cloud"]},
        {"type": "vEPC",       "cpu": (2, 4),  "ram": (2, 8),  "intensity": 1.0, "vcr": 1.0, "tiers": ["central_cloud"]},
    ],
}

# Per-slice-type QoS sampling ranges.
#
# `beta_in` is the slice's ingress data rate β_in (Mbps). The C5b throughput
# floor and the per-flow demands β_{k,k+1} are BOTH derived from β_in via
# v4 Eq. 3: β_{k,k+1} = β_in · ∏_{j=1}^{k} ρ_{f_j}. There is no independent
# "throughput" requirement in the formulation — that would be a free variable
# the model does not have. Older versions of this profile drew throughput
# independently of beta_in; that bug let slices be born with
# min_throughput > β_{k,k+1}, making C5b structurally unsatisfiable for
# compressing VCRs (∏ρ < 1) and silently breaking acceptance-rate eval.
_QOS_PROFILES: dict[SliceType, dict] = {
    SliceType.EMBB:  {"delay": (20.0, 100.0),  "beta_in": (50.0, 500.0)},
    SliceType.URLLC: {"delay": (1.0, 10.0),    "beta_in": (10.0, 50.0)},
    SliceType.MMTC:  {"delay": (50.0, 500.0),  "beta_in": (1.0, 10.0)},
    SliceType.V2X:   {"delay": (5.0, 20.0),    "beta_in": (20.0, 80.0)},
    SliceType.XR:    {"delay": (5.0, 30.0),    "beta_in": (100.0, 500.0)},
}

_SLICE_TYPE_WEIGHTS = {
    SliceType.EMBB:  0.30,
    SliceType.URLLC: 0.25,
    SliceType.MMTC:  0.20,
    SliceType.V2X:   0.15,
    SliceType.XR:    0.10,
}


def _resolve_permitted_nodes(
    tier_names: list[str],
    substrate: SubstrateNetwork,
) -> list[str]:
    """Return substrate node IDs matching any of the given tier names."""
    return [
        n for n, d in substrate.graph.nodes(data=True)
        if d["tier"] in tier_names
    ]


def generate_slice_request(
    request_id: str,
    substrate: SubstrateNetwork,
    rng: np.random.Generator,
    slice_type: SliceType | None = None,
    arrival_time: float = 0.0,
    lifetime: float = 0.0,
) -> SliceRequest:
    """Generate a single random slice request.

    Args:
        request_id: Unique request identifier.
        substrate: Substrate network for resolving permitted nodes (C8).
        rng: Seeded random generator.
        slice_type: If None, sample from weighted distribution.
        arrival_time: Simulation arrival time.
        lifetime: Duration; 0.0 for static batch mode.

    Returns:
        A fully populated SliceRequest.
    """
    if slice_type is None:
        types = list(_SLICE_TYPE_WEIGHTS.keys())
        weights = [_SLICE_TYPE_WEIGHTS[t] for t in types]
        # rng.choice returns a numpy element (numpy.str_ for StrEnum); coerce
        # back to a real SliceType so downstream `.value` access works.
        slice_type = SliceType(rng.choice(types, p=weights))

    templates = _VNF_TEMPLATES[slice_type]
    qos_profile = _QOS_PROFILES[slice_type]

    # Optionally shorten the chain (minimum 2 VNFs)
    max_vnfs = len(templates)
    n_vnfs = rng.integers(2, max_vnfs + 1) if max_vnfs > 2 else max_vnfs
    selected_templates = templates[:n_vnfs]

    vnfs: list[VNF] = []
    for k, tmpl in enumerate(selected_templates):
        cpu = float(rng.uniform(*tmpl["cpu"]))
        ram = float(rng.uniform(*tmpl["ram"]))
        permitted = _resolve_permitted_nodes(tmpl["tiers"], substrate)
        vnfs.append(VNF(
            vnf_id=f"{request_id}_f{k}",
            vnf_type=tmpl["type"],
            cpu_demand=round(cpu, 1),
            ram_demand=round(ram, 1),
            permitted_nodes=permitted,
            computational_intensity=tmpl["intensity"],
            vcr=tmpl["vcr"],
        ))

    # Single source of truth for the slice's bandwidth: the ingress rate β_in.
    # Per-edge flow demands AND the C5b throughput floor are both derived
    # from it. This is what v4 Eq. 3 prescribes.
    beta_in = float(rng.uniform(*qos_profile["beta_in"]))

    flow_edges: list[FlowEdge] = []
    for k in range(len(vnfs) - 1):
        # β_{k,k+1} = β_in · ∏_{j=1}^{k+1} ρ_{f_j}  (v4 Eq. 3)
        vcr_product = 1.0
        for j in range(k + 1):
            vcr_product *= vnfs[j].vcr
        bw = beta_in * vcr_product
        flow_edges.append(FlowEdge(
            source_vnf=vnfs[k].vnf_id,
            target_vnf=vnfs[k + 1].vnf_id,
            bandwidth_demand=round(bw, 1),
        ))

    delay = float(rng.uniform(*qos_profile["delay"]))

    return SliceRequest(
        request_id=request_id,
        slice_type=slice_type,
        vnfs=vnfs,
        flow_edges=flow_edges,
        qos=QoSRequirements(
            max_e2e_delay=round(delay, 1),
            # min_throughput IS β_in. C5b is satisfied iff every flow's
            # per-link allocation ≥ its derived β_{k,k+1} demand; the
            # verifier checks under-allocation, not an independent threshold.
            min_throughput=round(beta_in, 1),
        ),
        arrival_time=arrival_time,
        lifetime=lifetime,
    )


def generate_slice_batch(
    substrate: SubstrateNetwork,
    rng: np.random.Generator,
    num_requests: int,
) -> list[SliceRequest]:
    """Generate a static batch of slice requests (for MILP oracle evaluation).

    All requests have arrival_time=0.0 and lifetime=0.0.
    """
    return [
        generate_slice_request(
            request_id=f"req_{i:04d}",
            substrate=substrate,
            rng=rng,
        )
        for i in range(num_requests)
    ]
