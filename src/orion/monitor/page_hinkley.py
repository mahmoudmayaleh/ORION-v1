"""Page-Hinkley change detector.

Standard form for binary-outcome streams (rejection vs. acceptance):

    m_T  = (1/T) Σ_{i=1}^{T} x_i               (running mean)
    U_T  = Σ_{i=1}^{T} (x_i − m_T − δ)         (cumulative drift, lower-bounded by 0)
    PH_T = U_T − min_{1≤i≤T} U_i               (gap to historic minimum)
    drift detected when PH_T > h

`δ` is the drift tolerance (a small allowance to absorb noise), `h` is the
detection threshold. Online, O(1) memory per stream. Selected over CUSUM
and ADWIN because it has the simplest single-threshold calibration and is
the standard choice in streaming-RL literature for binary outcomes.

Defaults (δ=0.005, λ≈50 → h≈50·δ=0.25) follow the plan; tune on the
training arrival distribution and report in the ablation grid.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    """One-dimensional Page-Hinkley detector for a binary outcome stream.

    Attributes:
        delta: Drift tolerance. Excursions smaller than this are ignored.
        threshold: Detection threshold h. Higher → fewer false positives.
        min_instances: Burn-in length. Detection is suppressed until the
            running mean has stabilised across this many samples. Standard
            PH implementations (e.g., the River library) gate this way.
            Default 30 matches River's default and the plan's intent —
            without it, an early outlier produces a large transient in the
            PH statistic and trips the detector spuriously.
        count: Number of samples observed so far.
        running_mean: m_T.
        cumulative: U_T.
        min_cumulative: min over history of U_i.
    """

    delta: float = 0.005
    # `threshold` here is the λ parameter in the PH literature. The plan
    # specifies λ ≈ 50 as the default (NOT λ × δ — that was a misread). The
    # River library uses the same parameterisation and the same default.
    threshold: float = 50.0
    min_instances: int = 30
    count: int = 0
    running_mean: float = 0.0
    cumulative: float = 0.0
    min_cumulative: float = 0.0

    def update(self, x: float) -> bool:
        """Feed one observation. Returns True iff drift was detected this step."""
        self.count += 1
        # Incremental mean update.
        self.running_mean += (x - self.running_mean) / self.count
        # Drift accumulator.
        self.cumulative += x - self.running_mean - self.delta
        if self.cumulative < self.min_cumulative:
            self.min_cumulative = self.cumulative
        if self.count < self.min_instances:
            return False
        return (self.cumulative - self.min_cumulative) > self.threshold

    def reset(self) -> None:
        """Reset all state. Call after a detected drift is acted upon."""
        self.count = 0
        self.running_mean = 0.0
        self.cumulative = 0.0
        self.min_cumulative = 0.0
