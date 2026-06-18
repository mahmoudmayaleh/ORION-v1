"""Tests for MDO pre-commit verification."""

import pytest

from orion.actors.types import DomainResponse
from orion.mdo.precommit_check import (
    check_c5b_inter,
    compute_e2e_delay,
    compute_inter_domain_bw,
    compute_total_cost,
    count_inter_domain_hops,
    domain_sequence_from_partition,
    inter_domain_demand_by_pair,
    inter_domain_residual_by_pair,
    precommit_check,
)
from orion.mdo.types import InterDomainLink
from orion.types import QoSRequirements


class TestDomainSequence:
    def test_simple(self):
        assert domain_sequence_from_partition([0, 0, 1, 1, 2]) == [0, 1, 2]

    def test_alternating(self):
        assert domain_sequence_from_partition([0, 1, 0, 1]) == [0, 1, 0, 1]

    def test_single_domain(self):
        assert domain_sequence_from_partition([2, 2, 2]) == [2]

    def test_empty(self):
        assert domain_sequence_from_partition([]) == []


class TestInterDomainHops:
    def test_no_hops(self):
        assert count_inter_domain_hops([0, 0, 0]) == 0

    def test_two_hops(self):
        assert count_inter_domain_hops([0, 1, 2]) == 2

    def test_alternating(self):
        assert count_inter_domain_hops([0, 1, 0]) == 2


class TestInterDomainBW:
    def test_no_cross_domain(self):
        assert compute_inter_domain_bw([0, 0, 0], [100.0, 100.0]) == 0.0

    def test_cross_domain(self):
        bw = compute_inter_domain_bw([0, 1, 2], [100.0, 80.0])
        assert bw == pytest.approx(180.0)

    def test_partial_cross(self):
        bw = compute_inter_domain_bw([0, 0, 1], [100.0, 80.0])
        assert bw == pytest.approx(80.0)


class TestE2EDelay:
    def test_single_domain(self):
        responses = {0: DomainResponse(domain_id=0, feasible=True, intra_delay=5.0)}
        e2e = compute_e2e_delay(responses, {}, [0])
        assert e2e == pytest.approx(5.0)

    def test_multi_domain(self):
        responses = {
            0: DomainResponse(domain_id=0, feasible=True, intra_delay=3.0),
            1: DomainResponse(domain_id=1, feasible=True, intra_delay=4.0),
        }
        delays = {(0, 1): 2.0}
        e2e = compute_e2e_delay(responses, delays, [0, 1])
        assert e2e == pytest.approx(9.0)  # 3 + 4 + 2


class TestTotalCost:
    def test_intra_only(self):
        responses = {0: DomainResponse(domain_id=0, feasible=True, resource_cost=10.0)}
        cost = compute_total_cost(responses, inter_domain_bw=0.0)
        assert cost == pytest.approx(10.0)

    def test_with_inter(self):
        responses = {0: DomainResponse(domain_id=0, feasible=True, resource_cost=10.0)}
        cost = compute_total_cost(responses, inter_domain_bw=50.0, gamma_inter=2.0)
        assert cost == pytest.approx(110.0)  # 10 + 2*50


class TestPrecommitCheck:
    def test_all_pass(self):
        responses = {
            0: DomainResponse(domain_id=0, feasible=True, intra_delay=2.0, resource_cost=5.0),
            1: DomainResponse(domain_id=1, feasible=True, intra_delay=3.0, resource_cost=8.0),
        }
        qos = QoSRequirements(max_e2e_delay=20.0, min_throughput=50.0)
        passes, violation, e2e, cost = precommit_check(
            partition=[0, 0, 1],
            domain_responses=responses,
            inter_domain_delays={(0, 1): 2.0},
            qos=qos,
            bw_demands=[100.0, 80.0],
        )
        assert passes is True
        assert not violation.has_violation

    def test_c7_violated(self):
        responses = {
            0: DomainResponse(domain_id=0, feasible=True, intra_delay=15.0),
        }
        qos = QoSRequirements(max_e2e_delay=10.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 0],
            domain_responses=responses,
            inter_domain_delays={},
            qos=qos,
            bw_demands=[100.0],
        )
        assert passes is False
        assert violation.c7_violated is True

    def test_c9_violated(self):
        responses = {
            0: DomainResponse(domain_id=0, feasible=True, intra_delay=1.0),
            1: DomainResponse(domain_id=1, feasible=True, intra_delay=1.0),
            2: DomainResponse(domain_id=2, feasible=True, intra_delay=1.0),
            3: DomainResponse(domain_id=3, feasible=True, intra_delay=1.0),
        }
        qos = QoSRequirements(max_e2e_delay=100.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 1, 2, 3],
            domain_responses=responses,
            inter_domain_delays={(0,1): 1.0, (1,2): 1.0, (2,3): 1.0},
            qos=qos,
            bw_demands=[100.0, 80.0, 60.0],
            max_inter_domain_hops=2,
        )
        assert passes is False
        assert violation.c9_violated is True

    def test_actor_infeasible(self):
        responses = {
            0: DomainResponse(domain_id=0, feasible=False),
        }
        qos = QoSRequirements(max_e2e_delay=100.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 0],
            domain_responses=responses,
            inter_domain_delays={},
            qos=qos,
            bw_demands=[100.0],
        )
        assert passes is False
        assert violation.actor_infeasible is True


class TestInterDomainDemandByPair:
    def test_no_cross_domain(self):
        assert inter_domain_demand_by_pair([0, 0, 0], [10.0, 10.0]) == {}

    def test_single_cross(self):
        assert inter_domain_demand_by_pair([0, 1, 1], [60.0, 20.0]) == {(0, 1): 60.0}

    def test_orientation_normalised(self):
        # 1 -> 0 must key to (0, 1), same as 0 -> 1.
        assert inter_domain_demand_by_pair([1, 0], [40.0]) == {(0, 1): 40.0}

    def test_shared_pair_sums(self):
        # partition 0->1->0: both chain edges traverse the (0,1) pair; their
        # demands must sum so the shared inter-domain link sees joint load.
        assert inter_domain_demand_by_pair([0, 1, 0], [60.0, 60.0]) == {(0, 1): 120.0}


class TestInterDomainResidualByPair:
    def test_aggregates_both_orientations(self):
        links = [
            InterDomainLink(source_domain=0, target_domain=1, bw_residual=70.0,
                            bw_capacity=100.0, propagation_delay=2.0),
            InterDomainLink(source_domain=1, target_domain=0, bw_residual=30.0,
                            bw_capacity=100.0, propagation_delay=2.0),
        ]
        assert inter_domain_residual_by_pair(links) == {(0, 1): 100.0}

    def test_distinct_pairs(self):
        links = [
            InterDomainLink(source_domain=0, target_domain=1, bw_residual=50.0,
                            bw_capacity=100.0, propagation_delay=2.0),
            InterDomainLink(source_domain=1, target_domain=2, bw_residual=80.0,
                            bw_capacity=100.0, propagation_delay=2.0),
        ]
        assert inter_domain_residual_by_pair(links) == {(0, 1): 50.0, (1, 2): 80.0}


class TestCheckC5bInter:
    def test_pass(self):
        assert check_c5b_inter({(0, 1): 60.0}, {(0, 1): 100.0}) is False

    def test_violate(self):
        assert check_c5b_inter({(0, 1): 120.0}, {(0, 1): 100.0}) is True

    def test_missing_pair_is_violation(self):
        # Demand on a pair with no inter-domain residual entry = no capacity.
        assert check_c5b_inter({(0, 2): 10.0}, {(0, 1): 100.0}) is True

    def test_empty_demand_passes(self):
        assert check_c5b_inter({}, {(0, 1): 0.0}) is False


class TestPrecommitC5bInter:
    """Part A: aggregate per-pair inter-domain C5b in precommit_check.

    These inject depleted residuals directly — the live path runs against
    capacity until Part B's reservation lands, so the violation branch is
    only reachable from tests until then. This is the regression guard for
    the aggregate-per-pair logic.
    """

    def _feasible_responses(self):
        return {
            0: DomainResponse(domain_id=0, feasible=True, intra_delay=1.0),
            1: DomainResponse(domain_id=1, feasible=True, intra_delay=1.0),
        }

    def test_ample_residual_passes(self):
        qos = QoSRequirements(max_e2e_delay=100.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 1],
            domain_responses=self._feasible_responses(),
            inter_domain_delays={(0, 1): 1.0},
            qos=qos,
            bw_demands=[60.0],
            inter_domain_residuals={(0, 1): 100.0},
        )
        assert violation.c5b_violated is False
        assert passes is True

    def test_depleted_residual_violates(self):
        qos = QoSRequirements(max_e2e_delay=100.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 1],
            domain_responses=self._feasible_responses(),
            inter_domain_delays={(0, 1): 1.0},
            qos=qos,
            bw_demands=[60.0],
            inter_domain_residuals={(0, 1): 40.0},  # below the 60 demand
        )
        assert violation.c5b_violated is True
        assert passes is False

    def test_shared_pair_joint_overflow_violates(self):
        # partition 0->1->0: two cross-domain flows of 60 each on pair (0,1).
        # Each is under the 100 residual; jointly (120) they overflow. The
        # per-edge alternative would wrongly pass this.
        qos = QoSRequirements(max_e2e_delay=100.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 1, 0],
            domain_responses=self._feasible_responses(),
            inter_domain_delays={(0, 1): 1.0, (1, 0): 1.0},
            qos=qos,
            bw_demands=[60.0, 60.0],
            inter_domain_residuals={(0, 1): 100.0},
        )
        assert violation.c5b_violated is True
        assert passes is False

    def test_none_residuals_skips_check(self):
        # Parity with the pre-Part-A stub: no residual source => no C5b check,
        # even on a partition that would otherwise overflow.
        qos = QoSRequirements(max_e2e_delay=100.0, min_throughput=50.0)
        passes, violation, _, _ = precommit_check(
            partition=[0, 1],
            domain_responses=self._feasible_responses(),
            inter_domain_delays={(0, 1): 1.0},
            qos=qos,
            bw_demands=[9999.0],
            inter_domain_residuals=None,
        )
        assert violation.c5b_violated is False
        assert passes is True
