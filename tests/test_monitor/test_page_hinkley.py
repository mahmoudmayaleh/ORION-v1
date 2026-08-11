"""Page-Hinkley detector tests."""

from __future__ import annotations

from orion.monitor.page_hinkley import PageHinkley


class TestNoDriftStream:
    def test_low_steady_rate_does_not_trigger(self) -> None:
        ph = PageHinkley(delta=0.005, threshold=0.25)
        # Steady rejection rate of 10% for 200 samples — should NOT trigger.
        for i in range(200):
            triggered = ph.update(0.1 if i % 10 == 0 else 0.0)
            assert not triggered


class TestDrift:
    def test_sudden_rate_increase_triggers(self) -> None:
        ph = PageHinkley(delta=0.005, threshold=0.25)
        # 100 samples at 10% reject rate
        for i in range(100):
            ph.update(0.1 if i % 10 == 0 else 0.0)
        # then a sudden run of rejections
        triggered = False
        for _ in range(60):
            if ph.update(1.0):
                triggered = True
                break
        assert triggered, "PH must detect a clear regime shift to all-reject"


class TestReset:
    def test_reset_clears_state(self) -> None:
        ph = PageHinkley()
        for _ in range(50):
            ph.update(1.0)
        ph.reset()
        assert ph.count == 0
        assert ph.cumulative == 0.0
        assert ph.min_cumulative == 0.0
        assert ph.running_mean == 0.0
