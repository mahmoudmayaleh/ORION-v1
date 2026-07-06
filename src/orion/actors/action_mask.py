"""Action masking for domain actor VNF placement.

Computes a boolean mask over domain nodes indicating valid placement targets
for a given VNF. Enforces hard constraints only:
  - Tier compatibility (C8, belt-and-suspenders with structural checker)
  - Resource feasibility: residual CPU >= demand, residual RAM >= demand
  - Placement rules: node_id in VNF's permitted_nodes set

Soft constraints (C2/C3 load-balancing headroom) stay in the reward signal
so the policy learns to leave room for future arrivals.
"""

from __future__ import annotations

import torch

from orion.actors.types import VNFAssignment
from orion.substrate.graph_model import SubstrateNetwork
from orion.types import InfrastructureTier


def compute_action_mask(
    substrate: SubstrateNetwork,
    domain_node_ids: list[str],
    vnf: VNFAssignment,
    resource_overrides: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Compute a boolean mask over domain nodes for placing a VNF.

    Args:
        substrate: Current substrate state (digital twin).
        domain_node_ids: Ordered list of node IDs in this domain
            (must match the order used in domain_observation).
        vnf: The VNF assignment to place.
        resource_overrides: Optional dict mapping node_id to
            (remaining_cpu, remaining_ram) for autoregressive updates
            within a single fragment (before committing to substrate).

    Returns:
        Boolean tensor of shape [N_m] where True = valid placement target.
    """
    if resource_overrides is None:
        resource_overrides = {}

    permitted_set = set(vnf.permitted_nodes)
    n_nodes = len(domain_node_ids)
    mask = torch.zeros(n_nodes, dtype=torch.bool)

    for i, nid in enumerate(domain_node_ids):
        # C8: permitted nodes (includes tier check from slice generator)
        if nid not in permitted_set:
            continue

        # C8: explicit tier check (defense-in-depth)
        node_tier = InfrastructureTier(substrate.graph.nodes[nid]["tier"])
        if node_tier != vnf.required_tier:
            continue

        # Resource feasibility against current or overridden residuals
        if nid in resource_overrides:
            cpu_res, ram_res = resource_overrides[nid]
        else:
            cpu_res = substrate.get_residual_cpu(nid)
            ram_res = substrate.get_residual_ram(nid)

        if cpu_res < vnf.cpu_demand or ram_res < vnf.ram_demand:
            continue

        mask[i] = True

    return mask
