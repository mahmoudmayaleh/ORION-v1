"""Final reward assembly: take MDO components + ground-truth verdict → R_t.

The MDO coordinator (Phase 4) returns `RewardComponents` with admission,
efficiency, quality_shaping, and trial_penalty already correctly computed,
but stubs `hard_penalty = 0.0` because firing it requires the simulator-side
ground-truth check — that's what we add here (Choice E1).

This module is intentionally thin. It does NOT recompute admission, cost,
LocalScore, or trial counting; those belong to the MDO. It only:
  1. Overrides `hard_penalty` from the GroundTruthVerdict.
  2. Returns the final scalar R_t and the updated RewardComponents.
"""

from __future__ import annotations

from dataclasses import dataclass

from orion.mdo.types import RewardComponents
from orion.sim.verifier import GroundTruthVerdict


@dataclass(frozen=True)
class RewardWeights:
    """Reward term weights (v6.2 Eq. 9). Defaults match MDOConfig.

    These should be a single source of truth shared with MDOConfig in any
    production deployment — duplicated here as a frozen dataclass because
    the assembler may run independently of the MDO (e.g., during eval).
    """

    mu: float = 100.0
    alpha: float = 1.0
    lambda_viol: float = 10.0
    eta: float = 1.0
    xi: float = 0.5


def finalize_reward(
    mdo_components: RewardComponents,
    admitted: bool,
    verdict: GroundTruthVerdict | None,
    weights: RewardWeights | None = None,
) -> tuple[float, RewardComponents]:
    """Combine MDO-side components with the ground-truth hard-penalty check.

    Args:
        mdo_components: What the MDO coordinator returned for this arrival.
        admitted: Whether the slice was COMMITted.
        verdict: Ground-truth verdict from `verify_committed_plan`. Pass None
            when the slice was rejected (no plan to verify); the hard penalty
            stays at zero in that case.
        weights: Reward weights. Defaults to v6.2 Eq. 9 standard values.

    Returns:
        (R_t, RewardComponents) — the final scalar reward and the decomposed
        components (with hard_penalty filled in from the verdict).

    Notes:
        Hard penalty fires whenever the ground-truth check flags C2, C3, C5b,
        or C7 — including slices gated out by the verifier (admitted was True
        at commit but the verifier rolled it back to False). MDO-rejected
        slices have verdict=None so they never take the penalty.
    """
    w = weights or RewardWeights()

    hard_penalty = 0.0
    if verdict is not None and verdict.hard_penalty_fired:
        hard_penalty = -w.lambda_viol

    admission = mdo_components.admission if admitted else 0.0
    efficiency = mdo_components.efficiency if admitted else 0.0
    quality_shaping = mdo_components.quality_shaping if admitted else 0.0

    finalised = RewardComponents(
        admission=admission,
        efficiency=efficiency,
        hard_penalty=hard_penalty,
        quality_shaping=quality_shaping,
        trial_penalty=mdo_components.trial_penalty,
    )
    return finalised.total, finalised
