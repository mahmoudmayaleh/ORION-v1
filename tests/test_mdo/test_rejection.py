"""Tests for MDO rejection triggers."""

from orion.mdo.rejection import (
    check_budget_exhaustion,
    check_low_value_confidence,
    check_rejection_triggers,
    check_violation_stability,
)
from orion.mdo.types import (
    PartitionAttempt,
    RejectReason,
    RetryHistory,
    ViolationInfo,
)


class TestBudgetExhaustion:
    def test_not_exhausted(self):
        h = RetryHistory(attempts=[PartitionAttempt(0, [0])])
        assert check_budget_exhaustion(h, n_part=3) is False

    def test_exhausted(self):
        h = RetryHistory(attempts=[PartitionAttempt(i, [0]) for i in range(3)])
        assert check_budget_exhaustion(h, n_part=3) is True


class TestViolationStability:
    def test_no_stability(self):
        v1 = ViolationInfo(c7_violated=True)
        v2 = ViolationInfo(c9_violated=True)
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], violation=v1),
            PartitionAttempt(1, [1], violation=v2),
        ])
        assert check_violation_stability(h, k=2) is False

    def test_stable_violations(self):
        v1 = ViolationInfo(c7_violated=True)
        v2 = ViolationInfo(c7_violated=True)
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], violation=v1),
            PartitionAttempt(1, [1], violation=v2),
        ])
        assert check_violation_stability(h, k=2) is True

    def test_insufficient_attempts(self):
        v = ViolationInfo(c7_violated=True)
        h = RetryHistory(attempts=[PartitionAttempt(0, [0], violation=v)])
        assert check_violation_stability(h, k=2) is False


class TestLowValueConfidence:
    def test_all_below(self):
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], value_estimate=-5.0),
            PartitionAttempt(1, [1], value_estimate=-3.0),
        ])
        assert check_low_value_confidence(h, tau_v=0.0) is True

    def test_one_above(self):
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], value_estimate=-5.0),
            PartitionAttempt(1, [1], value_estimate=2.0),
        ])
        assert check_low_value_confidence(h, tau_v=0.0) is False

    def test_empty_history(self):
        h = RetryHistory()
        assert check_low_value_confidence(h, tau_v=0.0) is False


class TestCheckRejectionTriggers:
    def test_budget_fires_first(self):
        h = RetryHistory(attempts=[PartitionAttempt(i, [0]) for i in range(3)])
        result = check_rejection_triggers(h, n_part=3)
        assert result == RejectReason.BUDGET_EXHAUSTED

    def test_stability_fires(self):
        v = ViolationInfo(c7_violated=True)
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], violation=v),
            PartitionAttempt(1, [1], violation=ViolationInfo(c7_violated=True)),
        ])
        result = check_rejection_triggers(h, n_part=5, stability_k=2)
        assert result == RejectReason.VIOLATION_STABLE

    def test_no_trigger(self):
        v1 = ViolationInfo(c7_violated=True)
        v2 = ViolationInfo(c9_violated=True)
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], violation=v1, value_estimate=5.0),
            PartitionAttempt(1, [1], violation=v2, value_estimate=3.0),
        ])
        result = check_rejection_triggers(h, n_part=5)
        assert result is None

    def test_low_value_fires(self):
        h = RetryHistory(attempts=[
            PartitionAttempt(0, [0], value_estimate=-10.0),
            PartitionAttempt(1, [1], value_estimate=-8.0),
        ])
        result = check_rejection_triggers(h, n_part=5, tau_v=0.0)
        assert result == RejectReason.LOW_VALUE
