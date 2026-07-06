"""Strategy monitor tests — per-plan and global PH streams."""

from __future__ import annotations

from orion.monitor.strategy_monitor import StrategyMonitor


class TestPerPlanStream:
    def test_signature_specific_drift_marks_only_that_plan_stale(self) -> None:
        """The drift signal must be specific to the signature that drifted.

        PH detects *change*, not absolute level. We build a baseline of
        low rejection on `target`, then drive a rate increase, while
        `other` stays steady throughout.
        """
        m = StrategyMonitor(delta=0.005, threshold=0.1)
        target = ("eMBB", "loose")
        other = ("URLLC", "tight")

        # Baseline: 50 low-rejection samples on both signatures.
        for i in range(50):
            m.observe(target, rejected=(i % 20 == 0))   # ~5% reject baseline
            m.observe(other, rejected=(i % 20 == 0))    # same baseline

        # Drift: target rate spikes to 100%, other stays at baseline.
        stale_seen: list[tuple[str, str]] = []
        for i in range(80):
            m.observe(other, rejected=(i % 20 == 0))    # still 5%
            s = m.observe(target, rejected=True)        # now 100%
            stale_seen.extend(s.stale_plans)

        assert target in stale_seen
        assert other not in stale_seen


class TestGlobalStream:
    def test_global_drift_requests_refresh(self) -> None:
        m = StrategyMonitor(delta=0.005, threshold=0.1)
        seen_refresh = False
        for i in range(120):
            rejected = i > 50  # after step 50, everything rejects
            sig = ("eMBB", "loose")
            s = m.observe(sig, rejected=rejected)
            if s.global_refresh_requested:
                seen_refresh = True
                break
        assert seen_refresh


class TestNoFalsePositives:
    def test_steady_low_rejection_does_not_trigger(self) -> None:
        """Realistic default threshold (λ=50 per the plan / River library
        convention) must not flag a steady 5% baseline."""
        m = StrategyMonitor(delta=0.005, threshold=50.0)
        for i in range(300):
            rejected = (i % 20 == 0)  # 5% steady reject rate
            s = m.observe(("eMBB", "loose"), rejected=rejected)
            assert not s.stale_plans
            assert not s.global_refresh_requested
