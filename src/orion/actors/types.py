"""Data types for domain actor networks.

PlanFragment: the MDO-assigned subset of an abstract plan for one domain.
DomainResponse: the actor's output (feasibility, placements, routes, RL signals).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from orion.types import FlowEdge, InfrastructureTier


@dataclass
class VNFAssignment:
    """A single VNF assigned to this domain by the MDO partitioning."""

    vnf_id: str
    vnf_type: str
    cpu_demand: float
    ram_demand: float
    required_tier: InfrastructureTier
    computational_intensity: float
    vcr: float
    bandwidth_in: float
    permitted_nodes: list[str]
    position_in_sfc: int
    sfc_length: int
    adjacent_domain_ids: set[int] = field(default_factory=set)


@dataclass
class PlanFragment:
    """Portion of Agent B's abstract plan assigned to one domain by the MDO.

    Contains the VNFs to place, intra-domain flows to route, and cross-domain
    connectivity hints for border-node awareness.
    """

    domain_id: int
    vnf_assignments: list[VNFAssignment]
    intra_flows: list[FlowEdge]
    delay_budget_ms: float
    target_domain_ids: set[int] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return len(self.vnf_assignments) == 0


@dataclass
class ActorStepRecord:
    """Per-VNF placement inputs for PPO re-evaluation (CTDE).

    Stored during collection so the PPO update can re-evaluate the action
    under the current policy. graph_data and tensors are detached — they
    are replay inputs, not part of the gradient graph.
    """

    graph_data: object                # PyG Data (detached)
    vnf_context: torch.Tensor         # [VNF_CONTEXT_DIM]
    action_mask: torch.Tensor         # [N] boolean
    action_idx: int                   # chosen node (or DomainPolicy.NULL_ACTION)
    log_prob: float                   # scalar log-prob at collection time
    entropy: float


@dataclass
class DomainResponse:
    """Output of a domain actor after placement and routing.

    The MDO collects responses from all involved domains to compute E2E delay,
    check C5b/C7/C9, and decide COMMIT/RETRY.
    """

    domain_id: int
    feasible: bool
    placements: dict[str, str] = field(default_factory=dict)
    routes: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    bw_allocated: dict[tuple[str, str], float] = field(default_factory=dict)
    intra_delay: float = 0.0
    resource_cost: float = 0.0
    actions: list[int] = field(default_factory=list)
    log_probs: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    entropy: float = 0.0
    step_records: list[ActorStepRecord] = field(default_factory=list)

    @staticmethod
    def empty(domain_id: int) -> DomainResponse:
        """Trivial response when the MDO assigns zero VNFs to this domain."""
        return DomainResponse(domain_id=domain_id, feasible=True)
