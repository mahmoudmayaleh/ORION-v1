"""Tests for MDO observation construction."""

import torch

from orion.mdo.observation import build_tier_masks, observation_to_tensor
from orion.types import InfrastructureTier


class TestObservationToTensor:
    def test_returns_1d_tensor(self, simple_obs):
        t = observation_to_tensor(simple_obs)
        assert t.dim() == 1
        assert t.dtype == torch.float32

    def test_values_in_range(self, simple_obs):
        t = observation_to_tensor(simple_obs)
        # Most normalized features should be in [0, 1] or close
        assert t.isfinite().all()


class TestTierMasks:
    def test_shape(self, simple_plan, three_domain_summaries):
        mask = build_tier_masks(simple_plan, three_domain_summaries)
        assert mask.shape == (3, 3)  # 3 VNFs, 3 domains

    def test_edge_vnf_is_infeasible_in_the_cloud_only_domain(
            self, simple_plan, three_domain_summaries):
        mask = build_tier_masks(simple_plan, three_domain_summaries)
        # VNF 0 requires edge: d0 (edge-only) and d1 (edge+regional) support it,
        # d2 (cloud-only) does not.
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is True
        assert mask[0, 2].item() is False

    def test_second_edge_vnf_masks_identically_to_the_first(
            self, simple_plan, three_domain_summaries):
        mask = build_tier_masks(simple_plan, three_domain_summaries)
        # VNF 1 also requires edge, so its row must match VNF 0s exactly.
        assert mask[1, 0].item() is True
        assert mask[1, 1].item() is True
        assert mask[1, 2].item() is False

    def test_central_cloud_only_in_domain_2(self, simple_plan, three_domain_summaries):
        mask = build_tier_masks(simple_plan, three_domain_summaries)
        # VNF 2 requires CENTRAL_CLOUD — only domain 2 supports it
        assert mask[2, 0].item() is False
        assert mask[2, 1].item() is False
        assert mask[2, 2].item() is True

    def test_at_least_one_feasible_per_vnf(self, simple_plan, three_domain_summaries):
        mask = build_tier_masks(simple_plan, three_domain_summaries)
        for k in range(mask.shape[0]):
            assert mask[k].any(), f"VNF {k} has no feasible domain"
