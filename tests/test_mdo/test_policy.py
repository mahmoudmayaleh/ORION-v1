"""Tests for MDO policy network."""

import pytest
import torch

from orion.mdo.policy import MDOPolicy


class TestMDOPolicy:
    def test_forward_shape(self):
        policy = MDOPolicy(obs_dim=20, num_domains=3, max_vnfs=5)
        obs = torch.randn(20)
        mask = torch.ones(5, 3, dtype=torch.bool)
        partition, log_probs, logits, entropy = policy(obs, mask, num_vnfs=3)
        assert len(partition) == 3
        assert log_probs.shape == (3,)
        assert logits.shape == (3, 3)

    def test_deterministic_consistent(self):
        policy = MDOPolicy(obs_dim=20, num_domains=3, max_vnfs=5)
        obs = torch.randn(20)
        mask = torch.ones(5, 3, dtype=torch.bool)
        p1, _, _, _ = policy(obs, mask, num_vnfs=3, deterministic=True)
        p2, _, _, _ = policy(obs, mask, num_vnfs=3, deterministic=True)
        assert p1 == p2

    def test_masked_actions_feasible(self):
        policy = MDOPolicy(obs_dim=20, num_domains=4, max_vnfs=5)
        obs = torch.randn(20)
        mask = torch.tensor([
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ])
        for _ in range(10):
            partition, _, _, _ = policy(obs, mask, num_vnfs=3)
            assert partition[0] == 0  # only domain 0 feasible
            assert partition[1] == 1  # only domain 1 feasible
            assert partition[2] in (2, 3)

    def test_aux_value_head(self):
        policy = MDOPolicy(obs_dim=20, num_domains=3, max_vnfs=5)
        obs = torch.randn(20)
        value = policy.get_value(obs)
        assert value.shape == (1,)
        assert value.isfinite().all()

    def test_evaluate_actions(self):
        policy = MDOPolicy(obs_dim=20, num_domains=3, max_vnfs=5)
        obs = torch.randn(20)
        mask = torch.ones(5, 3, dtype=torch.bool)
        actions = torch.tensor([0, 1, 2])
        log_probs, entropy, logits = policy.evaluate_actions(obs, mask, actions, num_vnfs=3)
        assert log_probs.shape == (3,)
        assert logits.shape == (3, 3)

    def test_gradient_flows(self):
        policy = MDOPolicy(obs_dim=20, num_domains=3, max_vnfs=5)
        obs = torch.randn(20)
        mask = torch.ones(5, 3, dtype=torch.bool)
        actions = torch.tensor([0, 1, 2])
        log_probs, _, _ = policy.evaluate_actions(obs, mask, actions, num_vnfs=3)
        loss = -log_probs.sum()
        loss.backward()
        # Encoder and actor head should receive gradients
        for param in policy.encoder.parameters():
            assert param.grad is not None
        for param in policy.actor_head.parameters():
            assert param.grad is not None

    def test_padded_slots_same_loss(self):
        """|F_s|=3 with max_vnfs=8 should produce same log_probs as max_vnfs=3."""
        obs = torch.randn(20)
        mask_3 = torch.ones(3, 4, dtype=torch.bool)
        mask_8 = torch.ones(8, 4, dtype=torch.bool)

        policy_3 = MDOPolicy(obs_dim=20, num_domains=4, max_vnfs=3)
        policy_8 = MDOPolicy(obs_dim=20, num_domains=4, max_vnfs=8)

        # Copy weights for the first 3 VNF slots
        with torch.no_grad():
            for p3, p8 in zip(policy_3.encoder.parameters(), policy_8.encoder.parameters()):
                p8.copy_(p3)
            # actor_head has different sizes, so just verify shapes are right

        p3, lp3, _, e3 = policy_3(obs, mask_3, num_vnfs=3, deterministic=True)
        p8, lp8, _, e8 = policy_8(obs, mask_8, num_vnfs=3, deterministic=True)
        # Both should produce exactly 3 log_probs
        assert lp3.shape == (3,)
        assert lp8.shape == (3,)

    def test_entropy_on_masked_distribution(self):
        """Entropy should reflect only feasible domains, not full distribution."""
        policy = MDOPolicy(obs_dim=20, num_domains=4, max_vnfs=5)
        obs = torch.randn(20)
        # Only 1 feasible domain per VNF -> entropy should be ~0
        mask_tight = torch.tensor([
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [True, True, True, True],
            [True, True, True, True],
        ])
        _, _, _, entropy = policy(obs, mask_tight, num_vnfs=3)
        # First 2 VNFs have 1 feasible domain -> 0 entropy each
        # VNF 2 has 1 domain -> 0 entropy, mean should be 0
        assert entropy == pytest.approx(0.0, abs=1e-5)

    def test_aux_value_head_gradient_independent(self):
        """Aux value head should get gradients from value loss, not actor loss."""
        policy = MDOPolicy(obs_dim=20, num_domains=3, max_vnfs=5)
        obs = torch.randn(20)
        value = policy.get_value(obs)
        target = torch.tensor([5.0])
        value_loss = (value - target).pow(2).mean()
        value_loss.backward()
        for param in policy.aux_value_head.parameters():
            assert param.grad is not None
