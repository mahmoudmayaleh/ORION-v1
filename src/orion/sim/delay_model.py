"""Load-dependent delay model: M/M/1 sojourn times for nodes and links.

This is the ground-truth delay model used by the post-commit verifier
(`sim/verifier.py`). The MDO pre-commit check uses the static light-load
approximation; the gap between the two is what fires the C7 hard penalty
in the reward (v6.2 Section 3.2).

Form (textbook M/M/1):

    sojourn(load) = base_delay × intensity + 1 / (μ − load)

Where μ is the service rate (CPU capacity for nodes, bandwidth capacity
for links) and `load` is the offered load (sum of allocated demand).
When load ≥ μ the queue is saturated and sojourn → ∞.

Reference: Kleinrock, *Queueing Systems Vol. 1*, ch. 3. RouteNet-Fermi
[arXiv:2212.12070] is cited as the higher-fidelity GNN-based alternative
for future work.
"""

from __future__ import annotations

import math

_EPS = 1e-9


def mm1_sojourn(
    base_delay: float,
    intensity: float,
    service_rate: float,
    load: float,
) -> float:
    """M/M/1 sojourn time with computational-intensity scaling.

    δ(load) = base_delay × intensity + 1 / (μ − load)

    Args:
        base_delay: Static delay floor (processing_delay for nodes, propagation
            delay for links), in ms.
        intensity: Computational/throughput intensity multiplier (1.0 for links,
            VNF.computational_intensity for nodes).
        service_rate: μ — service rate (CPU capacity / bandwidth capacity).
        load: Current offered load on this resource.

    Returns:
        Sojourn time in ms. +inf if load ≥ μ (saturated queue).
    """
    headroom = service_rate - load
    if headroom <= _EPS:
        return math.inf
    return base_delay * intensity + 1.0 / headroom


def node_sojourn(
    base_processing_delay: float,
    intensity: float,
    cpu_capacity: float,
    cpu_used: float,
) -> float:
    """Sojourn time at a node hosting one or more VNFs.

    Service rate μ = cpu_capacity; load = cpu_used (cpu_capacity − cpu_residual).
    """
    return mm1_sojourn(
        base_delay=base_processing_delay,
        intensity=intensity,
        service_rate=cpu_capacity,
        load=cpu_used,
    )


def link_sojourn(
    propagation_delay: float,
    bandwidth_capacity: float,
    bandwidth_used: float,
) -> float:
    """Sojourn time on a link carrying one or more flows.

    Service rate μ = bandwidth_capacity (Mbps); load = bandwidth_used.
    Intensity is 1.0 for links — only nodes carry intensity scaling.
    """
    return mm1_sojourn(
        base_delay=propagation_delay,
        intensity=1.0,
        service_rate=bandwidth_capacity,
        load=bandwidth_used,
    )
