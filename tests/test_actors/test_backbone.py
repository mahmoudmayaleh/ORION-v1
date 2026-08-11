"""Tests for GNN backbone architectures."""

from __future__ import annotations

import torch

from orion.actors.backbone import GATv2Backbone, GatedGCNBackbone, MLPBackbone
from orion.actors.domain_observation import build_domain_observation


class TestGATv2Backbone:

    def test_forward_shape(self, small_substrate):
        """Output shape should be [N, hidden_dim]."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        backbone = GATv2Backbone(hidden_dim=128, num_heads=4)

        out = backbone(data)
        assert out.shape == (len(node_ids), 128)

    def test_gradient_flow(self, small_substrate):
        """Gradients should flow through all layers."""
        data, _ = build_domain_observation(small_substrate, domain_id=0)
        backbone = GATv2Backbone(hidden_dim=64, num_heads=2)

        out = backbone(data)
        loss = out.sum()
        loss.backward()

        for name, param in backbone.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_deterministic_with_seed(self, small_substrate):
        """Same seed should give same output."""
        data, _ = build_domain_observation(small_substrate, domain_id=0)
        backbone = GATv2Backbone(hidden_dim=64, num_heads=2)

        torch.manual_seed(42)
        out1 = backbone(data)
        torch.manual_seed(42)
        out2 = backbone(data)

        assert torch.allclose(out1, out2)


class TestGatedGCNBackbone:

    def test_forward_shape(self, small_substrate):
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        backbone = GatedGCNBackbone(hidden_dim=128)

        out = backbone(data)
        assert out.shape == (len(node_ids), 128)

    def test_gradient_flow(self, small_substrate):
        data, _ = build_domain_observation(small_substrate, domain_id=0)
        backbone = GatedGCNBackbone(hidden_dim=64)

        out = backbone(data)
        loss = out.sum()
        loss.backward()

        for name, param in backbone.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMLPBackbone:

    def test_forward_shape(self, small_substrate):
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)
        backbone = MLPBackbone(hidden_dim=128)

        out = backbone(data)
        assert out.shape == (len(node_ids), 128)

    def test_ignores_graph_structure(self, small_substrate):
        """MLP output should not change if we permute edge_index."""
        data, _ = build_domain_observation(small_substrate, domain_id=0)
        backbone = MLPBackbone(hidden_dim=64)

        out1 = backbone(data)

        # Permute edges
        import copy
        data2 = copy.deepcopy(data)
        if data2.edge_index.shape[1] > 1:
            perm = torch.randperm(data2.edge_index.shape[1])
            data2.edge_index = data2.edge_index[:, perm]
            data2.edge_attr = data2.edge_attr[perm]

        out2 = backbone(data2)
        assert torch.allclose(out1, out2)
