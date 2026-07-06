"""Tests for MDO data types."""

from orion.mdo.types import (
    MDOAction,
    RejectReason,
    RewardComponents,
    RetryHistory,
    ViolationInfo,
    PartitionAttempt,
    PlanSummary,
)
from orion.types import InfrastructureTier


class TestMDOAction:
    def test_enum_values(self):
        assert MDOAction.COMMIT == 0
        assert MDOAction.RETRY == 1
        assert MDOAction.REJECT == 2

    def test_reject_reason_values(self):
        assert RejectReason.BUDGET_EXHAUSTED == 0
        assert RejectReason.VIOLATION_STABLE == 1
        assert RejectReason.LOW_VALUE == 2


class TestRewardComponents:
    def test_total(self):
        r = RewardComponents(admission=100.0, efficiency=-5.0, hard_penalty=-10.0, quality_shaping=2.0, trial_penalty=-1.0)
        assert r.total == 86.0

    def test_zero_default(self):
        r = RewardComponents()
        assert r.total == 0.0


class TestRetryHistory:
    def test_empty(self):
        h = RetryHistory()
        assert h.num_attempts == 0
        assert h.last_violation_vectors(2) == []

    def test_violation_vectors(self):
        v1 = ViolationInfo(c7_violated=True)
        v2 = ViolationInfo(c7_violated=True)
        h = RetryHistory(attempts=[
            PartitionAttempt(trial_index=0, partition=[0, 1], violation=v1),
            PartitionAttempt(trial_index=1, partition=[1, 0], violation=v2),
        ])
        vecs = h.last_violation_vectors(2)
        assert len(vecs) == 2
        assert vecs[0] == (False, True, False, False, False)

    def test_num_attempts(self):
        h = RetryHistory(attempts=[
            PartitionAttempt(trial_index=0, partition=[0]),
            PartitionAttempt(trial_index=1, partition=[1]),
        ])
        assert h.num_attempts == 2


class TestPlanSummary:
    def test_num_vnfs(self, simple_plan):
        assert simple_plan.num_vnfs == 3


class TestViolationInfo:
    def test_violation_vector(self):
        v = ViolationInfo(c5b_violated=True, c9_violated=True)
        assert v.violation_vector == (True, False, True, False, False)
        assert v.has_violation is True

    def test_no_violation(self):
        v = ViolationInfo()
        assert v.has_violation is False
