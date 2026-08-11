"""Data types for the MDO RL coordinator.

MDOObservation: aggregated cross-domain state for the MDO policy.
PartitionDecision: record of the partition decision for an arrival.
MDOAction: COMMIT / REJECT enum.
MDOResult: final outcome per slice arrival.
RewardComponents: decomposed reward for logging and debugging.
StrategyMonitorState: types for the population-level Page-Hinkley monitor.
PlanCacheEntry: cached abstract plan with staleness tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

import torch

from orion.actors.types import DomainResponse
from orion.types import InfrastructureTier, SliceType


class MDOAction(IntEnum):
    """Control action emitted by the MDO after collecting domain responses."""
    COMMIT = 0
    REJECT = auto()


class RejectReason(IntEnum):
    """Why the arrival was rejected."""
    INFEASIBLE = 0


@dataclass
class DomainSummary:
    """Per-domain aggregated state for the MDO observation.

    Sorted by canonical key (tier_type, domain_id) for stable ordering.
    """
    domain_id: int
    dominant_tier: InfrastructureTier
    cpu_residual: float
    ram_residual: float
    cpu_capacity: float
    ram_capacity: float
    supported_tiers: list[InfrastructureTier] = field(default_factory=list)
    active_slice_count: int = 0
    # Single-node fragmentation headroom (PREREG 2026-07-11 §M.4-Δ3): the best-fitting node's
    # min(cpu_res/c_ref, ram_res/r_ref) over nodes in the domain. Aggregate residuals hide
    # whether ANY single node is large enough; this exposes it. 0.0 = no node has headroom.
    max_node_headroom: float = 0.0
    # Residual CPU/RAM PER TIER (§Y.1e). Aggregate residuals cannot express "this
    # domain's edge tier is full but its regional tier is not", which is the common
    # case once domains hold different tier sets. Every tier is always a key,
    # including tiers this domain does not hold, which read 0.0: an absent tier and
    # an exhausted tier are deliberately indistinguishable, since composition is
    # fixed and an absent tier reads 0.0 in every instance.
    tier_cpu_residual: dict[InfrastructureTier, float] = field(default_factory=dict)
    tier_ram_residual: dict[InfrastructureTier, float] = field(default_factory=dict)


@dataclass
class InterDomainLink:
    """Aggregated inter-domain link for the MDO observation."""
    source_domain: int
    target_domain: int
    bw_residual: float
    bw_capacity: float
    propagation_delay: float


@dataclass
class PlanSummary:
    """Flattened representation of Agent B's abstract plan for the MDO.

    Contains per-VNF tier requirements, suggested domain assignments,
    resource demands, and the suggested partition vector m̃.
    """
    vnf_ids: list[str]
    required_tiers: list[InfrastructureTier]
    suggested_domains: list[int]
    cpu_demands: list[float]
    ram_demands: list[float]
    vcrs: list[float]
    bw_demands: list[float]  # per-flow edge bandwidth requirements

    @property
    def num_vnfs(self) -> int:
        return len(self.vnf_ids)


@dataclass
class MDOObservation:
    """Aggregated cross-domain observation for the MDO policy (v6.2 Eq. 2).

    o^MDO_t = ({ĉ^m_res, r̂^m_res, τ^m}, {b^res_ℓ, D_ℓ}, π̃_t)
    """
    domain_summaries: list[DomainSummary]
    inter_domain_links: list[InterDomainLink]
    plan: PlanSummary

    @property
    def num_domains(self) -> int:
        return len(self.domain_summaries)


@dataclass
class ViolationInfo:
    """Structured violation information from a failed partition decision."""
    c5b_violated: bool = False
    c7_violated: bool = False
    c9_violated: bool = False
    actor_infeasible: bool = False  # any z^m = 0
    cross_domain_infeasible: bool = False
    e2e_delay: float = 0.0
    e2e_budget: float = 0.0
    total_bw: float = 0.0
    min_bw: float = 0.0
    inter_domain_hops: int = 0
    max_inter_domain_hops: int = 0

    @property
    def violation_vector(self) -> tuple[bool, ...]:
        """(C5b, C7, C9, actor_infeasible, cross_domain_infeasible) for stability detection."""
        return (
            self.c5b_violated, self.c7_violated, self.c9_violated,
            self.actor_infeasible, self.cross_domain_infeasible,
        )

    @property
    def has_violation(self) -> bool:
        return any(self.violation_vector)


@dataclass
class PartitionDecision:
    """Record of the partition decision for an arrival."""
    partition: list[int]  # per-VNF domain assignment
    domain_responses: dict[int, DomainResponse] = field(default_factory=dict)
    violation: ViolationInfo | None = None
    e2e_delay: float = 0.0
    total_cost: float = 0.0
    value_estimate: float = 0.0  # V^MDO_ψ output
    log_probs: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    entropy: float = 0.0


@dataclass
class RewardComponents:
    """Decomposed reward (v6.2 Eq. 9) for logging and debugging.

    R_t = μ·z_s - α·Cost(π*) - λ_viol·1[violated] + η·LocalScore(π*)
    """
    admission: float = 0.0        # μ·z_s
    efficiency: float = 0.0       # -α·Cost(π*)
    hard_penalty: float = 0.0     # -λ_viol·1[C2,C3,C5b,C7 violated]
    quality_shaping: float = 0.0  # +η·LocalScore(π*)

    @property
    def total(self) -> float:
        return (
            self.admission
            + self.efficiency
            + self.hard_penalty
            + self.quality_shaping
        )


@dataclass
class MDOResult:
    """Final outcome of the MDO coordinator for one slice arrival."""
    request_id: str
    action: MDOAction
    admitted: bool
    partition: list[int] | None = None  # final committed partition
    domain_responses: dict[int, DomainResponse] = field(default_factory=dict)
    cross_domain_routes: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    cross_domain_bw: dict[tuple[str, str], float] = field(default_factory=dict)
    e2e_delay: float = 0.0
    total_cost: float = 0.0
    reward: RewardComponents = field(default_factory=RewardComponents)
    decision: PartitionDecision | None = None
    reject_reason: RejectReason | None = None
    # §Y.13 — why an already-COMMITTED admission was revoked.
    #
    # `resolve_arrival` returns admitted=True and the episode runner then
    # allocates, verifies against post-allocation load, and may revoke. Before
    # this field existed that flip recorded nothing the rejection taxonomy could
    # read: `_classify` inspects `decision.violation`, which is None on this
    # path, so 100% of post-commit revocations landed in `unattributed` (24 to 89
    # percent of all rejections). Holds the post-commit verifier's violated
    # constraint codes, e.g. ["C7"], or ["PLAN_BUILD"] when no concrete
    # placement could be synthesised from the committed partition.
    #
    # Codes rather than the GroundTruthVerdict itself: the verdict lives in
    # `orion.sim.verifier` and this module is `orion.mdo`, so holding the object
    # would point a model type at the simulator that consumes it.
    revoked_by: list[str] | None = None
    log_probs: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    entropy: float = 0.0
    value_estimate: float = 0.0
    obs_tensor: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    tier_mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    num_vnfs: int = 0


# --- Population-level strategy monitor types (logic deferred to Phase 5) ---

@dataclass
class PlanCacheEntry:
    """Cached abstract plan with staleness tracking."""
    signature: str  # slice-spec signature (service_type + qos_tier_bucket)
    plan: dict[str, Any]  # the cached π̃ (abstract plan dict)
    suggested_partition: list[int] | None = None
    stale: bool = False
    resolution_count: int = 0
    rejection_count: int = 0


@dataclass
class StrategyMonitorState:
    """State for the Page-Hinkley change detector (per-plan or global).

    Types defined here; detector logic deferred to Phase 5.
    """
    running_mean: float = 0.0
    cumulative_stat: float = 0.0  # G_i
    sample_count: int = 0
    drift_tolerance: float = 0.005  # δ
    threshold: float = 50.0  # h_P or h_G
