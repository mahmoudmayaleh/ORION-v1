"""Tests for MDO data types."""

from orion.mdo.types import (
    MDOAction,
    RejectReason,
    RewardComponents,
    ViolationInfo,
    PlanSummary,
)
from orion.types import InfrastructureTier


class TestMDOAction:
    def test_enum_values(self):
        assert MDOAction.COMMIT == 0
        assert MDOAction.REJECT == 1

    def test_reject_reason_values(self):
        # INFEASIBLE is the only reason left: VIOLATION_STABLE and LOW_VALUE
        # belonged to the multi-attempt coordinator that the single-attempt
        # directive removed.
        assert list(RejectReason) == [RejectReason.INFEASIBLE]
        assert RejectReason.INFEASIBLE == 0


class TestRewardComponents:
    def test_total(self):
        r = RewardComponents(admission=100.0, efficiency=-5.0, hard_penalty=-10.0, quality_shaping=2.0)
        assert r.total == 87.0

    def test_zero_default(self):
        r = RewardComponents()
        assert r.total == 0.0


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
