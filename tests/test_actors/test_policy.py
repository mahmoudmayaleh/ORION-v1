"""Tests for domain placement policy (pointer network with NULL/refuse token)."""

from __future__ import annotations

import torch

from orion.actors.domain_observation import build_domain_observation
from orion.actors.policy import DomainPolicy, VNF_CONTEXT_DIM


class TestDomainPolicy:

    def _make_vnf_context(self) -> torch.Tensor:
        """Dummy VNF context vector."""
        ctx = torch.zeros(VNF_CONTEXT_DIM)
        ctx[0] = 0.5  # cpu_demand_norm
        ctx[1] = 0.3  # ram_demand_norm
        ctx[3] = 1.0  # tier: mec
        ctx[6] = 1.0  # vcr
        ctx[7] = 0.2  # bw_demand_norm
        ctx[8] = 0.0  # position_in_sfc_norm
        return ctx

    def test_output_types(self, small_substrate):
        """forward should return (int, Tensor, float)."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        mask = torch.ones(len(node_ids), dtype=torch.bool)

        torch.manual_seed(42)
        action, log_prob, entropy = policy(data, self._make_vnf_context(), mask)

        assert isinstance(action, int)
        # Action is either a valid node index or NULL_ACTION (-1)
        assert action == DomainPolicy.NULL_ACTION or (0 <= action < len(node_ids))
        assert isinstance(log_prob, torch.Tensor)
        assert log_prob.ndim == 0  # scalar
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_masked_nodes_never_selected(self, small_substrate):
        """Actions should only be from unmasked nodes or NULL."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)

        # Mask all but the last node
        mask = torch.zeros(len(node_ids), dtype=torch.bool)
        mask[-1] = True

        for _ in range(10):
            action, _, _ = policy(data, self._make_vnf_context(), mask)
            # Should be either the last node or NULL
            assert action == len(node_ids) - 1 or action == DomainPolicy.NULL_ACTION

    def test_deterministic_mode(self, small_substrate):
        """Deterministic mode should give consistent results."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        mask = torch.ones(len(node_ids), dtype=torch.bool)
        ctx = self._make_vnf_context()

        action1, _, _ = policy(data, ctx, mask, deterministic=True)
        action2, _, _ = policy(data, ctx, mask, deterministic=True)
        assert action1 == action2

    def test_all_false_mask_selects_null(self, small_substrate):
        """All-False mask forces selection of the NULL/refuse token."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        mask = torch.zeros(len(node_ids), dtype=torch.bool)

        action, log_prob, entropy = policy(data, self._make_vnf_context(), mask)
        assert action == DomainPolicy.NULL_ACTION
        # With only one valid slot (NULL), log_prob should be 0 and entropy 0
        assert abs(log_prob.item()) < 1e-5
        assert abs(entropy) < 1e-5

    def test_probabilities_sum_to_one(self, small_substrate):
        """Softmax output over [N+1] should sum to ~1.0."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        mask = torch.ones(len(node_ids), dtype=torch.bool)
        ctx = self._make_vnf_context()

        # Use _encode_and_score to get the full [N+1] logit vector
        scores = policy._encode_and_score(data, ctx, mask)
        probs = torch.softmax(scores, dim=0)

        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert probs.size(0) == len(node_ids) + 1  # N real + 1 NULL

    def test_gradient_flows_through_policy(self, small_substrate):
        """Gradients should reach backbone parameters through log_prob."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        mask = torch.ones(len(node_ids), dtype=torch.bool)

        _, log_prob, _ = policy(data, self._make_vnf_context(), mask)
        (-log_prob).backward()

        has_grad = False
        for name, param in policy.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients reached any parameter"

    def test_null_is_unreachable_while_any_node_is_feasible(self, small_substrate):
        """The NULL slot is masked whenever a real placement exists.

        Contract CHANGED 2026-08-20 (RL_DIAGNOSIS §9). NULL used to be
        unconditionally valid, which handed PPO a free "give up" action the
        reward pays for: a coordinator-level reject scores 0 while an
        admitted-then-revoked arrival scores -1, so refusing strictly dominates
        attempting a placement that might miss the delay budget. Measured, the
        actor learned exactly that -- 101 of 196 refusals in one segment were
        taken while a feasible node existed, and each one fails the arrival.
        Refusal is the coordinator's authority (MDOAction.COMMIT/REJECT).

        This test previously asserted the opposite (that a gradient reaches
        null_token with a fully-feasible mask). It is inverted, not deleted, so
        the change of contract is recorded rather than silently dropped.
        """
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)
        ctx = self._make_vnf_context()

        feasible = torch.ones(len(node_ids), dtype=torch.bool)
        scores = policy._encode_and_score(data, ctx, feasible)
        assert scores[-1] == float("-inf"), (
            "NULL is reachable while real nodes are feasible; PPO will learn "
            "to refuse rather than place")
        for _ in range(10):
            action, log_prob, _ = policy(data, ctx, feasible)
            assert action != DomainPolicy.NULL_ACTION
            assert log_prob.isfinite()

        # ... and it is still reachable, as the only option, when nothing fits.
        empty = torch.zeros(len(node_ids), dtype=torch.bool)
        scores = policy._encode_and_score(data, ctx, empty)
        assert torch.isfinite(scores[-1]), "NULL must remain the fallback"
        action, log_prob, _ = policy(data, ctx, empty)
        assert action == DomainPolicy.NULL_ACTION
        assert log_prob.isfinite()

    def test_all_backbone_types_work(self, small_substrate):
        """All three backbone types should produce valid output."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        mask = torch.ones(len(node_ids), dtype=torch.bool)
        ctx = self._make_vnf_context()

        for backbone_type in ["gatv2", "gatedgcn", "mlp"]:
            policy = DomainPolicy(backbone_type=backbone_type, hidden_dim=64, num_heads=2)
            action, log_prob, entropy = policy(data, ctx, mask)
            assert action == DomainPolicy.NULL_ACTION or (0 <= action < len(node_ids))
            assert log_prob.isfinite()

    def test_forward_bc_shape(self, small_substrate):
        """forward_bc should return [K, N+1] logits for K VNFs."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)

        k = 3
        vnf_contexts = torch.randn(k, VNF_CONTEXT_DIM)
        logits = policy.forward_bc(data, vnf_contexts)

        assert logits.shape == (k, len(node_ids) + 1)

    def test_forward_bc_gradient_flow(self, small_substrate):
        """BC forward should support gradient computation for supervised loss."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        policy = DomainPolicy(backbone_type="gatv2", hidden_dim=64, num_heads=2)

        k = 2
        vnf_contexts = torch.randn(k, VNF_CONTEXT_DIM)
        logits = policy.forward_bc(data, vnf_contexts)

        # Simulate cross-entropy loss with random labels
        labels = torch.randint(0, len(node_ids) + 1, (k,))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.parameters()
        )
        assert has_grad, "No gradients from BC loss"
