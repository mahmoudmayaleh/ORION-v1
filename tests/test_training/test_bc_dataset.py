"""BC dataset tests — Choice B1 reproducibility contract."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from orion.config import TopologyConfig
from orion.training.bc_dataset import (
    BCDatasetSpec,
    generate_dataset,
    get_or_generate_dataset,
    load_dataset,
    save_dataset,
)


@pytest.fixture
def spec() -> BCDatasetSpec:
    return BCDatasetSpec(
        seed=42,
        num_scenarios=10,
        topology_config=TopologyConfig(
            num_domains=3,
            nodes_per_domain=[5, 5, 5],
            intra_link_density=0.6,
            inter_domain_links=2,
        ),
    )


class TestReproducibility:
    def test_same_spec_yields_same_config_hash(self, spec) -> None:
        h1 = spec.config_hash()
        h2 = spec.config_hash()
        assert h1 == h2

    def test_different_seed_changes_hash(self, spec) -> None:
        h_a = spec.config_hash()
        spec.seed = 99
        h_b = spec.config_hash()
        assert h_a != h_b

    def test_different_num_scenarios_changes_hash(self, spec) -> None:
        h_a = spec.config_hash()
        spec.num_scenarios = 999
        h_b = spec.config_hash()
        assert h_a != h_b


class TestRoundTrip:
    def test_save_and_load(self, spec, tmp_path) -> None:
        samples = generate_dataset(spec)
        path = tmp_path / "bc.pt"
        h = save_dataset(samples, spec, path)
        assert path.exists()
        loaded, meta = load_dataset(path)
        assert meta["dataset_hash"] == h
        assert meta["config_hash"] == spec.config_hash()
        assert len(loaded) == len(samples)

    def test_get_or_generate_reuses_cache(self, spec, tmp_path) -> None:
        path = tmp_path / "bc.pt"
        s1, m1 = get_or_generate_dataset(spec, path)
        s2, m2 = get_or_generate_dataset(spec, path)
        assert m1["dataset_hash"] == m2["dataset_hash"]
        assert len(s1) == len(s2)


class TestConfigDrift:
    def test_config_change_invalidates_cache(self, spec, tmp_path) -> None:
        """If the topology config changes, the cached dataset must be
        silently regenerated — training on stale demonstrations is the
        failure mode B1 is meant to prevent."""
        path = tmp_path / "bc.pt"
        _, meta_a = get_or_generate_dataset(spec, path)
        spec.num_scenarios = 5  # different config
        _, meta_b = get_or_generate_dataset(spec, path)
        assert meta_a["config_hash"] != meta_b["config_hash"]
