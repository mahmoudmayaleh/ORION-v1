"""Scenario-class slice factories for the §exp transfer grid.

The paper's two workload regimes (Section~\\ref{sec:exp-setup}) differ only in the
per-VNF volume-change ratio (VCR, ρ), which controls how inter-VNF bandwidth evolves
along the SFC chain:

  * Conventional (ρ = 1.0): bandwidth is held constant along the chain,
    β_{k,k+1} = β_in — the setting most prior slicing work assumes.
  * Stress (ρ > 1): bandwidth GROWS along the chain, β_{k,k+1} = β_in · ρ^k,
    tightening the coupling between placement and inter-domain routing. ρ matches
    the validated pre-§Y routing-critical family (rho = 1.15). That generator was
    removed in §Y.1e; the VALUE is retained because the scenario class is defined by
    it, not by the generator it was measured on.

Both wrap the shipped ``generate_slice_request`` (same slice-type mix, chain
templates, QoS ranges, and permitted-node resolution) and override ONLY the VCR,
recomputing the bandwidth ramp from β_in exactly as v4 Eq. 3 prescribes. This keeps
the workload identical across topologies (slice-mix FROZEN
across families") while making the scenario class the single varying factor.

Both factories satisfy the SliceFactory contract
(request_id, substrate, rng, arrival_time, lifetime) -> SliceRequest, so they drop
into ``ArrivalProcess(..., slice_factory=...)`` and wp7_runner's ``RC_SLICE_FACTORY``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from orion.substrate.graph_model import SubstrateNetwork
from orion.sim.slice_generator import (
    COMPLEX_SLICE_TYPE_WEIGHTS,
    generate_slice_request,
)
from orion.types import FlowEdge, SliceRequest

# Stress ρ, matched to the validated RC family (_RC_VCR).
STRESS_RHO = 1.15


def _reramp(sr: SliceRequest, rho: float) -> SliceRequest:
    """Return a copy of ``sr`` with every VNF's VCR set to ``rho`` and the flow-edge
    bandwidths recomputed as β_{k,k+1} = β_in · ρ^{k+1} (β_in = qos.min_throughput,
    the ingress rate — the single source of truth per generate_slice_request)."""
    beta_in = sr.qos.min_throughput
    new_vnfs = [replace(v, vcr=rho) for v in sr.vnfs]
    new_edges = []
    for k in range(len(new_vnfs) - 1):
        bw = beta_in * (rho ** (k + 1))
        new_edges.append(FlowEdge(
            source_vnf=new_vnfs[k].vnf_id,
            target_vnf=new_vnfs[k + 1].vnf_id,
            bandwidth_demand=round(bw, 1),
        ))
    return replace(sr, vnfs=new_vnfs, flow_edges=new_edges)


def make_scenario_slice_factory(scenario: str, rho: float | None = None):
    """Build a family-agnostic SliceFactory for the given scenario class.

    scenario == "conventional" -> ρ = 1.0 (constant bandwidth).
    scenario == "stress"       -> ρ = STRESS_RHO (ramp), or an explicit ``rho``.
    scenario == "complex"      -> ρ = 1.0, XR-dominant mix (§Y.10 level S2).
    """
    weights = None
    if scenario == "conventional":
        r = 1.0
    elif scenario == "stress":
        r = STRESS_RHO if rho is None else rho
    elif scenario == "complex":
        # §Y.10 level S2. Same VCR as conventional, so this differs from the
        # reference in the workload MIX and in nothing else -- the one-axis-at-a-
        # time requirement of §Y.4. Chains get longer and heavier together
        # because XR chains ARE longer and heavier, not via a demand multiplier.
        r = 1.0
        weights = COMPLEX_SLICE_TYPE_WEIGHTS
    else:
        raise ValueError(f"unknown scenario class: {scenario!r}")

    def factory(request_id: str, substrate: SubstrateNetwork, rng: np.random.Generator,
                arrival_time: float = 0.0, lifetime: float = 0.0) -> SliceRequest:
        sr = generate_slice_request(request_id, substrate, rng,
                                    arrival_time=arrival_time, lifetime=lifetime,
                                    type_weights=weights)
        return _reramp(sr, r)

    factory.scenario = scenario  # type: ignore[attr-defined]
    factory.rho = r  # type: ignore[attr-defined]
    return factory
