"""β_t schedules for the KL-prior term in the MDO loss.

The KL prior pulls the MDO toward Agent B's suggested partition m̃. The
default schedule is linear decay (Choice D1): start high to leverage the
LLM's service-level knowledge, decay to zero so the MDO can outgrow it.

Ablation axis (all three live here, share the same MAPPO core, isolate
the prior's value):

    beta_zero      — β ≡ 0          (no prior at all)
    beta_constant  — β ≡ const      (prior never relaxes)
    beta_linear    — β decays       (default — outgrow the prior)

`target_kl_adaptive` is intentionally NOT in this file. That's a different
regulariser (pins the policy at a fixed divergence from Agent B forever,
the opposite of the design intent); placing it next to these would mis-
characterise the ablation.
"""

from __future__ import annotations


def beta_zero(_step: int) -> float:
    """No KL prior. Used to measure the prior's contribution from the floor."""
    return 0.0


def beta_constant(_step: int, value: float) -> float:
    """Constant β. Used to measure 'prior never relaxes' vs the decaying default."""
    return value


def beta_linear(
    step: int,
    beta_initial: float,
    beta_final: float,
    decay_steps: int,
) -> float:
    """Linear decay from `beta_initial` to `beta_final` over `decay_steps`.

    After `decay_steps`, the value is pinned at `beta_final` (typically 0).
    """
    if decay_steps <= 0:
        return beta_final
    if step >= decay_steps:
        return beta_final
    if step <= 0:
        return beta_initial
    progress = step / decay_steps
    return beta_initial + (beta_final - beta_initial) * progress
