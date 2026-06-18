"""Rejection triggers for the MDO partition retry loop.

Three triggers, evaluated after each failed partition trial (v6.2 Section 3.4):
  (i)   Budget exhaustion: T_t = N_part
  (ii)  Violation-signature stability: same constraint fails k≥2 trials
        with non-decreasing slack
  (iii) Low value confidence: V^MDO_ψ below τ_V for all partitions so far

The first trigger to fire ends the partition loop and rejects the slice.
"""

from __future__ import annotations

from orion.mdo.types import RejectReason, RetryHistory


def check_budget_exhaustion(
    history: RetryHistory,
    n_part: int,
) -> bool:
    """Trigger (i): partition budget exhausted."""
    return history.num_attempts >= n_part


def check_violation_stability(
    history: RetryHistory,
    k: int = 2,
) -> bool:
    """Trigger (ii): same constraint(s) fail across last k trials.

    If the same violation vector repeats k times in a row, the plan is
    structurally infeasible for the current substrate state regardless
    of partitioning. Reject early.

    Args:
        history: Retry history with accumulated attempts.
        k: Minimum consecutive identical violation vectors to trigger.

    Returns:
        True if the stability condition is met.
    """
    if history.num_attempts < k:
        return False

    vectors = history.last_violation_vectors(k)
    if len(vectors) < k:
        return False

    # Check if all last k vectors are identical and non-trivial
    first = vectors[0]
    if not any(first):  # no violations at all — should not happen here
        return False

    return all(v == first for v in vectors)


def check_low_value_confidence(
    history: RetryHistory,
    tau_v: float,
) -> bool:
    """Trigger (iii): V^MDO_ψ below threshold for all partitions so far.

    If the auxiliary value head estimates low value for every partition
    sampled, no high-value partition exists in the local neighbourhood.

    Args:
        history: Retry history with value estimates per attempt.
        tau_v: Calibrated value threshold.

    Returns:
        True if all attempts have value below threshold.
    """
    if history.num_attempts == 0:
        return False

    return all(
        attempt.value_estimate < tau_v
        for attempt in history.attempts
    )


def check_rejection_triggers(
    history: RetryHistory,
    n_part: int,
    tau_v: float = -float("inf"),
    stability_k: int = 2,
) -> RejectReason | None:
    """Evaluate all rejection triggers and return the first that fires.

    Args:
        history: Retry history for the current arrival.
        n_part: Maximum partition retry budget.
        tau_v: Value threshold for trigger (iii). Default -inf disables it
            (useful before the aux value head is trained).
        stability_k: Consecutive identical violations for trigger (ii).

    Returns:
        The RejectReason if a trigger fired, or None if RETRY is still valid.
    """
    if check_budget_exhaustion(history, n_part):
        return RejectReason.BUDGET_EXHAUSTED

    if check_violation_stability(history, stability_k):
        return RejectReason.VIOLATION_STABLE

    if check_low_value_confidence(history, tau_v):
        return RejectReason.LOW_VALUE

    return None
