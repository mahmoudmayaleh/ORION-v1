"""PPO update metrics + separate-scalar logging contract.

The hard requirement is: policy_loss, value_loss, entropy_bonus, and
kl_prior_term are returned as separate scalars in `PPOMetrics`. Fusing
them into a single regularisation term would break diagnostics; this
test exists to lock that contract in even if the optimisation logic
evolves.
"""

from __future__ import annotations

from orion.training.ppo_update import PPOMetrics, explained_variance


class TestPPOMetricsContract:
    def test_default_fields_zero(self) -> None:
        m = PPOMetrics()
        assert m.policy_loss == 0.0
        assert m.value_loss == 0.0
        assert m.entropy_bonus == 0.0
        assert m.kl_prior_term == 0.0
        assert m.approx_kl == 0.0
        assert m.clip_fraction == 0.0
        assert m.total_loss == 0.0

    def test_separate_scalar_fields_exist(self) -> None:
        # Pins the field names — anyone collapsing entropy + kl_prior into
        # a "regularisation" scalar will break this test.
        fields = PPOMetrics.__dataclass_fields__
        for name in (
            "policy_loss",
            "value_loss",
            "entropy_bonus",
            "kl_prior_term",
            "approx_kl",
            "clip_fraction",
            "total_loss",
        ):
            assert name in fields, f"missing required metric field: {name}"


class TestExplainedVariance:
    def test_perfect_fit_is_one(self) -> None:
        import torch
        r = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v = r.clone()
        assert explained_variance(r, v) == 1.0

    def test_mean_baseline_is_zero(self) -> None:
        import torch
        r = torch.tensor([1.0, 2.0, 3.0, 4.0])
        v = torch.full_like(r, r.mean().item())
        assert explained_variance(r, v) == 0.0

    def test_zero_variance_returns_zero_safely(self) -> None:
        import torch
        r = torch.zeros(5)
        v = torch.zeros(5)
        assert explained_variance(r, v) == 0.0
