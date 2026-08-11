"""Reward assembler tests."""

from __future__ import annotations

from orion.mdo.types import RewardComponents
from orion.sim.reward import RewardWeights, finalize_reward
from orion.sim.verifier import GroundTruthVerdict


def _components(
    admission: float = 0.0,
    efficiency: float = 0.0,
    quality_shaping: float = 0.0,
) -> RewardComponents:
    # `trial_penalty` was removed with the single-attempt coordinator: there is
    # one PartitionDecision per arrival and no retry, so there is no trial to
    # charge for. These tests carried it until 2026-08-03 and had been failing at
    # collection since.
    return RewardComponents(
        admission=admission,
        efficiency=efficiency,
        hard_penalty=0.0,  # always 0 from coordinator; reward layer overrides
        quality_shaping=quality_shaping,
    )


class TestHardPenaltyFire:
    def test_admitted_with_clean_verdict_no_penalty(self) -> None:
        verdict = GroundTruthVerdict(feasible=True, violated=[])
        r, comps = finalize_reward(
            _components(admission=100.0, efficiency=-10.0),
            admitted=True,
            verdict=verdict,
        )
        assert comps.hard_penalty == 0.0
        assert r == 90.0

    def test_admitted_with_c7_violation_fires_penalty(self) -> None:
        verdict = GroundTruthVerdict(feasible=False, violated=["C7"])
        r, comps = finalize_reward(
            _components(admission=100.0, efficiency=-10.0),
            admitted=True,
            verdict=verdict,
            weights=RewardWeights(lambda_viol=10.0),
        )
        assert comps.hard_penalty == -10.0
        assert r == 80.0

    def test_admitted_with_c5_only_does_not_fire(self) -> None:
        # C5 is not in {C2,C3,C5b,C7} so it must not trigger the hard penalty.
        verdict = GroundTruthVerdict(feasible=False, violated=["C5"])
        r, comps = finalize_reward(
            _components(admission=100.0),
            admitted=True,
            verdict=verdict,
        )
        assert comps.hard_penalty == 0.0
        assert r == 100.0

    def test_admitted_with_c9_only_does_not_fire(self) -> None:
        verdict = GroundTruthVerdict(feasible=False, violated=["C9"])
        _, comps = finalize_reward(
            _components(admission=100.0),
            admitted=True,
            verdict=verdict,
        )
        assert comps.hard_penalty == 0.0


class TestRejectedSlice:
    def test_rejected_with_no_verdict_no_penalty(self) -> None:
        r, comps = finalize_reward(
            _components(admission=0.0),
            admitted=False,
            verdict=None,
        )
        assert comps.hard_penalty == 0.0
        assert r == 0.0

    def test_verifier_gated_slice_gets_penalty_but_no_admission(self) -> None:
        verdict = GroundTruthVerdict(feasible=False, violated=["C7"])
        r, comps = finalize_reward(
            _components(admission=100.0),
            admitted=False,
            verdict=verdict,
        )
        assert comps.admission == 0.0
        assert comps.hard_penalty == -10.0
        assert r == -10.0


class TestVerifierGate:
    def test_gated_slice_zeros_admitted_components(self) -> None:
        verdict = GroundTruthVerdict(feasible=False, violated=["C7"])
        _, comps = finalize_reward(
            _components(admission=100.0, efficiency=-5.0, quality_shaping=2.0),
            admitted=False,
            verdict=verdict,
        )
        assert comps.admission == 0.0
        assert comps.efficiency == 0.0
        assert comps.quality_shaping == 0.0
        assert comps.hard_penalty == -10.0

    def test_mdo_rejected_no_verdict_no_penalty(self) -> None:
        _, comps = finalize_reward(
            _components(admission=0.0),
            admitted=False,
            verdict=None,
        )
        assert comps.hard_penalty == 0.0
        assert comps.admission == 0.0


class TestWeights:
    def test_custom_lambda(self) -> None:
        verdict = GroundTruthVerdict(feasible=False, violated=["C2"])
        _, comps = finalize_reward(
            _components(admission=100.0),
            admitted=True,
            verdict=verdict,
            weights=RewardWeights(lambda_viol=42.0),
        )
        assert comps.hard_penalty == -42.0
