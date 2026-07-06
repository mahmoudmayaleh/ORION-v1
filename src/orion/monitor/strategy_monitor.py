"""Population-level strategy monitor (v6.2 §6.5 / Phase 4 §4.8).

Two Page-Hinkley streams running in parallel:

    - Per-plan stream G^P: one PH detector per cache key. Triggers when
      THIS cached plan's rejection rate drifts up. Effect: mark the
      cache entry stale so the next matching arrival triggers a refresh.

    - Global stream G_i: one PH detector over all slice resolutions.
      Triggers when the global rejection rate drifts up. Effect: schedule
      an asynchronous Agent B strategy review with a summary of recent
      failure patterns.

Both effects are signals to the trainer (return values, not side effects).
The trainer decides when to act on them — e.g., batch refreshes between
PPO updates so the LLM call doesn't sit on the rollout hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orion.monitor.page_hinkley import PageHinkley


@dataclass
class MonitorSignals:
    """What the strategy monitor returns after each slice resolution."""

    stale_plans: list[tuple[str, str]] = field(default_factory=list)
    global_refresh_requested: bool = False


@dataclass
class StrategyMonitor:
    """Owns the per-plan and global PH detectors plus their thresholds."""

    delta: float = 0.005
    # `threshold` = λ (PH literature). λ ≈ 50 is the plan default; River
    # library uses the same parameterisation. Tests that want a sensitive
    # detector override this explicitly to a low value (e.g., 0.1).
    threshold: float = 50.0

    per_plan: dict[tuple[str, str], PageHinkley] = field(default_factory=dict)
    global_stream: PageHinkley = field(default_factory=lambda: PageHinkley())

    def __post_init__(self) -> None:
        # Re-init the global stream with the configured thresholds.
        self.global_stream = PageHinkley(delta=self.delta, threshold=self.threshold)

    def observe(
        self,
        signature: tuple[str, str],
        rejected: bool,
    ) -> MonitorSignals:
        """Feed one slice resolution. Returns triggered signals (possibly empty).

        `rejected` is 1 for any system-level rejection (structural OR MDO).
        Both contribute to the global stream; only the per-plan stream is
        keyed by the cache signature.
        """
        signals = MonitorSignals()
        x = 1.0 if rejected else 0.0

        # Per-plan stream.
        detector = self.per_plan.setdefault(
            signature, PageHinkley(delta=self.delta, threshold=self.threshold)
        )
        if detector.update(x):
            signals.stale_plans.append(signature)
            detector.reset()  # acted on; reset before next monitor cycle

        # Global stream.
        if self.global_stream.update(x):
            signals.global_refresh_requested = True
            self.global_stream.reset()

        return signals
