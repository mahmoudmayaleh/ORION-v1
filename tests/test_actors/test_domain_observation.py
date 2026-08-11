"""Tests for domain observation (PyG Data) construction."""

from __future__ import annotations

import torch

from orion.actors.domain_observation import (
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    build_domain_observation,
)


class TestBuildDomainObservation:

    def test_output_shapes(self, small_substrate):
        """Node features are [N, 15] and edge features are [E, 2]."""
        data, node_ids = build_domain_observation(small_substrate, domain_id=0)

        n_nodes = len(node_ids)
        assert data.x.shape == (n_nodes, NODE_FEAT_DIM)
        assert data.edge_attr.shape[1] == EDGE_FEAT_DIM
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == data.edge_attr.shape[0]

    def test_node_features_in_valid_range(self, small_substrate):
        """All normalized features should be in [0, 1]."""
        data, _ = build_domain_observation(small_substrate, domain_id=0)

        # Fraction features (0, 1, 2, 3, 8, 9, 10, 11) should be in [0, 1]
        for col in [0, 1, 2, 3, 8, 9, 10, 11]:
            assert data.x[:, col].min() >= 0.0, f"Feature {col} has negative values"
            assert data.x[:, col].max() <= 1.0 + 1e-6, f"Feature {col} exceeds 1.0"

        # One-hot tier columns (4-7) should be 0 or 1
        tier_cols = data.x[:, 4:8]
        assert ((tier_cols == 0) | (tier_cols == 1)).all()
        assert tier_cols.sum(dim=1).allclose(torch.ones(data.x.shape[0]))

        # Binary features (12, 13, 14) should be 0 or 1
        for col in [12, 13, 14]:
            assert ((data.x[:, col] == 0) | (data.x[:, col] == 1)).all()

    def test_edge_features_in_valid_range(self, small_substrate):
        """Edge BW frac and delay norm should be in [0, 1]."""
        data, _ = build_domain_observation(small_substrate, domain_id=0)

        if data.edge_attr.shape[0] > 0:
            assert data.edge_attr[:, 0].min() >= 0.0
            assert data.edge_attr[:, 0].max() <= 1.0 + 1e-6
            assert data.edge_attr[:, 1].min() >= 0.0
            assert data.edge_attr[:, 1].max() <= 1.0 + 1e-6

    def test_border_node_features(self, small_substrate):
        """Domain 0 should have border nodes connecting to domain 1."""
        data, node_ids = build_domain_observation(
            small_substrate, domain_id=0,
            target_domain_ids={1},
        )
        # At least one border node should exist (inter-domain links exist)
        border_flags = data.x[:, 12]
        assert border_flags.sum() > 0, "No border nodes detected in domain 0"

        # border_to_target should be <= border count
        target_flags = data.x[:, 13]
        assert target_flags.sum() <= border_flags.sum()

    def test_placed_vnf_feature(self, small_substrate):
        """has_placed_vnf feature should reflect placed_node_ids."""
        node_ids = small_substrate.nodes_in_domain(0)
        placed = {node_ids[0]}

        data, obs_ids = build_domain_observation(
            small_substrate, domain_id=0,
            placed_node_ids=placed,
        )

        idx = obs_ids.index(node_ids[0])
        assert data.x[idx, 14] == 1.0
        # Other nodes should be 0
        for i, nid in enumerate(obs_ids):
            if nid not in placed:
                assert data.x[i, 14] == 0.0

    def test_multiple_domains_independent(self, small_substrate):
        """Observations for different domains should have different node counts."""
        data0, ids0 = build_domain_observation(small_substrate, domain_id=0)
        data1, ids1 = build_domain_observation(small_substrate, domain_id=1)

        assert set(ids0) != set(ids1)
        assert set(ids0).isdisjoint(set(ids1))

    def test_node_ids_are_sorted(self, small_substrate):
        """Returned node_ids should be sorted for canonical ordering."""
        _, node_ids = build_domain_observation(small_substrate, domain_id=0)
        assert node_ids == sorted(node_ids)
