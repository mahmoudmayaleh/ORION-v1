"""Data types for the MDO RL coordinator.

MDOObservation: aggregated cross-domain state for the MDO policy.
PartitionAttempt: record of one partition trial within an arrival.
RetryHistory: accumulates attempts per arrival.
MDOAction: COMMIT / RETRY / REJECT enum.
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
    RETRY = auto()
    REJECT = auto()


class RejectReason(IntEnum):
    """Which rejection trigger fired."""
    BUDGET_EXHAUSTED = 0
    VIOLATION_STABLE = auto()
    LOW_VALUE = auto()


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

    o^MDO_t = ({ĉ^m_res, r̂^m_res, τ^m}, {b^res_ℓ, D_ℓ}, π̃_t, h_t)
    """
    domain_summaries: list[DomainSummary]
    inter_domain_links: list[InterDomainLink]
    plan: PlanSummary
    retry_history: RetryHistory | None = None

    @property
    def num_domains(self) -> int:
        return len(self.domain_summaries)


@dataclass
class ViolationInfo:
    """Structured violation information from a failed partition trial."""
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
class PartitionAttempt:
    """Record of one partition trial within an arrival."""
    trial_index: int
    partition: list[int]  # per-VNF domain assignment
    domain_responses: dict[int, DomainResponse] = field(default_factory=dict)
    violation: ViolationInfo | None = None
    e2e_delay: float = 0.0
    total_cost: float = 0.0
    value_estimate: float = 0.0  # V^MDO_ψ output
    log_probs: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    entropy: float = 0.0


@dataclass
class RetryHistory:
    """Accumulates partition attempts for one slice arrival."""
    attempts: list[PartitionAttempt] = field(default_factory=list)

    @property
    def num_attempts(self) -> int:
        return len(self.attempts)

    def last_violation_vectors(self, k: int = 2) -> list[tuple[bool, bool, bool, bool]]:
        """Return the violation vectors from the last k attempts."""
        recent = self.attempts[-k:]
        return [a.violation.violation_vector for a in recent if a.violation is not None]


@dataclass
class RewardComponents:
    """Decomposed reward (v6.2 Eq. 9) for logging and debugging.

    R_t = μ·z_s - α·Cost(π*) - λ_viol·1[violated] + η·LocalScore(π*) - ξ·(T_t-1)
    """
    admission: float = 0.0        # μ·z_s
    efficiency: float = 0.0       # -α·Cost(π*)
    hard_penalty: float = 0.0     # -λ_viol·1[C2,C3,C5b,C7 violated]
    quality_shaping: float = 0.0  # +η·LocalScore(π*)
    trial_penalty: float = 0.0    # -ξ·(T_t - 1)

    @property
    def total(self) -> float:
        return (
            self.admission
            + self.efficiency
            + self.hard_penalty
            + self.quality_shaping
            + self.trial_penalty
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
    retry_history: RetryHistory = field(default_factory=RetryHistory)
    reject_reason: RejectReason | None = None
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
