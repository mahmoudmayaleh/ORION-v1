"""Tests for analytical KL divergence and prior construction."""

import pytest
import torch
from torch.autograd import gradcheck

from orion.mdo.kl_prior import analytical_kl, build_prior_logits, beta_schedule


class TestAnalyticalKL:
    def test_identical_distributions_zero_kl(self):
        logits = torch.randn(3, 4)
        kl = analytical_kl(logits, logits.clone())
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_kl(self):
        logits_p = torch.tensor([[2.0, 0.0, -1.0]])
        logits_q = torch.tensor([[0.0, 0.0, 0.0]])
        kl = analytical_kl(logits_p, logits_q)
        assert kl.item() > 0.0

    def test_masked_kl(self):
        logits_p = torch.tensor([[1.0, 2.0, 3.0]])
        logits_q = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        kl = analytical_kl(logits_p, logits_q, mask=mask)
        assert kl.item() >= 0.0

    def test_single_feasible_domain_zero_kl(self):
        logits_p = torch.tensor([[5.0, -1.0, -1.0]])
        logits_q = torch.tensor([[0.0, -1.0, -1.0]])
        mask = torch.tensor([[True, False, False]])
        kl = analytical_kl(logits_p, logits_q, mask=mask)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flows(self):
        logits_p = torch.randn(2, 3, requires_grad=True)
        logits_q = torch.randn(2, 3)
        kl = analytical_kl(logits_p, logits_q)
        kl.backward()
        assert logits_p.grad is not None
        assert logits_p.grad.shape == (2, 3)

    def test_gradcheck_finite_difference(self):
        """Verify analytical gradient matches finite-difference approximation."""
        logits_p = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        logits_q = torch.randn(2, 3, dtype=torch.float64)
        assert gradcheck(analytical_kl, (logits_p, logits_q), eps=1e-6, atol=1e-4)

    def test_multi_vnf_slots(self):
        logits_p = torch.randn(5, 4)
        logits_q = torch.randn(5, 4)
        kl = analytical_kl(logits_p, logits_q)
        assert kl.item() >= 0.0

    def test_zero_feasible_domain_returns_zero(self):
        """All-False mask for a VNF slot should not produce NaN."""
        logits_p = torch.randn(2, 3)
        logits_q = torch.randn(2, 3)
        mask = torch.tensor([[True, True, True], [False, False, False]])
        kl = analytical_kl(logits_p, logits_q, mask=mask)
        assert torch.isfinite(kl)
        assert kl.item() >= 0.0

    def test_all_zero_mask_returns_zero(self):
        """Entirely infeasible mask should return 0."""
        logits_p = torch.randn(3, 4)
        logits_q = torch.randn(3, 4)
        mask = torch.zeros(3, 4, dtype=torch.bool)
        kl = analytical_kl(logits_p, logits_q, mask=mask)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_padded_slots_ignored(self):
        """KL on 3 real VNFs should equal KL with 5 padded slots (extra masked)."""
        logits_3 = torch.randn(3, 4)
        logits_q_3 = torch.randn(3, 4)
        kl_3 = analytical_kl(logits_3, logits_q_3)

        # Pad to 5 slots with all-False masks on slots 3,4
        logits_5 = torch.cat([logits_3, torch.randn(2, 4)])
        logits_q_5 = torch.cat([logits_q_3, torch.randn(2, 4)])
        mask_5 = torch.ones(5, 4, dtype=torch.bool)
        mask_5[3:] = False
        kl_5 = analytical_kl(logits_5, logits_q_5, mask=mask_5)

        assert kl_3.item() == pytest.approx(kl_5.item(), abs=1e-4)


class TestBuildPriorLogits:
    def test_shape(self):
        mask = torch.ones(3, 4, dtype=torch.bool)
        logits = build_prior_logits([0, 1, 2], num_domains=4, tier_masks=mask)
        assert logits.shape == (3, 4)

    def test_peak_on_suggested(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        logits = build_prior_logits([1, 2], num_domains=3, tier_masks=mask, temperature=0.1)
        probs = torch.softmax(logits, dim=-1)
        assert probs[0, 1].item() > probs[0, 0].item()
        assert probs[1, 2].item() > probs[1, 0].item()

    def test_infeasible_masked(self):
        mask = torch.tensor([[True, False, True], [False, True, True]])
        logits = build_prior_logits([0, 1], num_domains=3, tier_masks=mask)
        probs = torch.softmax(logits, dim=-1)
        assert probs[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert probs[1, 0].item() == pytest.approx(0.0, abs=1e-6)


class TestBetaSchedule:
    def test_start(self):
        assert beta_schedule(0, 1000, 1.0, 0.01) == pytest.approx(1.0)

    def test_end(self):
        assert beta_schedule(1000, 1000, 1.0, 0.01) == pytest.approx(0.01)

    def test_midpoint(self):
        mid = beta_schedule(500, 1000, 1.0, 0.0)
        assert mid == pytest.approx(0.5)

    def test_beyond_total(self):
        val = beta_schedule(2000, 1000, 1.0, 0.01)
        assert val == pytest.approx(0.01)
