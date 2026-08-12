"""Guards for the occupancy metrics added 2026-08-12.

The failure mode these exist for is silence. A sampler that is never called still
produces a summary; it just produces one built from zero samples, and a utilisation
of 0.0 reads as "this approach is efficient" rather than "this number was never
measured". So the hook is asserted to FIRE, once per arrival, and the summary is
asserted to move when the substrate does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cost_metrics import CostAccumulator  # noqa: E402
from orion.substrate.hierarchical_topology import (  # noqa: E402
    generate_hierarchical_topology,
)
from orion.types import TIER_ORDER  # noqa: E402


def test_an_empty_episode_reports_no_samples_not_zero_utilisation():
    """Never measured and measured-as-empty must not look the same."""
    acc = CostAccumulator(generate_hierarchical_topology(0))
    assert acc.utilization_summary() == {"n_samples": 0}


def test_utilisation_is_zero_on_an_untouched_substrate_and_rises_with_load():
    sub = generate_hierarchical_topology(0)
    acc = CostAccumulator(sub)
    acc.sample_utilization()
    first = acc.utilization_summary()
    assert first["n_samples"] == 1
    assert first["node_cpu_util_mean"] == 0.0
    assert first["node_ram_util_mean"] == 0.0
    assert first["inter_domain_bw_util"] == 0.0
    assert first["cpu_capacity"] > 0 and first["ram_capacity"] > 0

    # Consume half of one node. The mean must move by that node's share and no more.
    n = sorted(sub.graph.nodes)[0]
    cap = sub.graph.nodes[n]["cpu_capacity"]
    sub.graph.nodes[n]["cpu_residual"] = cap / 2.0
    acc.sample_utilization()
    second = acc.utilization_summary()
    assert second["n_samples"] == 2
    # Time-averaged over the two samples: 0 then (cap/2)/total.
    # The reported ratio is rounded to 4 dp, so compare at that resolution.
    expected = (cap / 2.0) / second["cpu_capacity"] / 2.0
    assert second["node_cpu_util_mean"] == pytest.approx(expected, abs=5e-5)
    assert second["cpu_allocated_mean"] == pytest.approx(cap / 4.0, rel=1e-3)


def test_every_reported_ratio_is_a_ratio():
    """Per-tier utilisation divides by that tier's own capacity, not the whole
    substrate's, or the central-cloud tier would read near-zero forever purely
    because it holds few nodes."""
    sub = generate_hierarchical_topology(0)
    for n in sub.graph.nodes:
        sub.graph.nodes[n]["cpu_residual"] = 0.0
        sub.graph.nodes[n]["ram_residual"] = 0.0
    acc = CostAccumulator(sub)
    acc.sample_utilization()
    s = acc.utilization_summary()
    assert s["node_cpu_util_mean"] == pytest.approx(1.0)
    assert s["node_ram_util_mean"] == pytest.approx(1.0)
    assert set(s["node_cpu_util_by_tier"]) == {t.value for t in TIER_ORDER}
    for v in s["node_cpu_util_by_tier"].values():
        assert v == pytest.approx(1.0)
    for v in s["node_ram_util_by_tier"].values():
        assert v == pytest.approx(1.0)
    assert s["node_cpu_frac_over_90"] == pytest.approx(1.0)


def test_link_utilisation_separates_inter_from_intra_domain():
    """Inter-domain bandwidth is the scarce resource the partition decision spends.
    If the two were pooled, a busy intra-domain fabric would mask an idle boundary
    and the metric would stop discriminating between partitions."""
    sub = generate_hierarchical_topology(0)
    acc = CostAccumulator(sub)
    assert acc._inter_edges and acc._intra_edges, "one of the two classes is empty"
    for u, v, _c in acc._inter_edges:
        sub.graph.edges[u, v]["bw_residual"] = 0.0
    acc.sample_utilization()
    s = acc.utilization_summary()
    assert s["inter_domain_bw_util"] == pytest.approx(1.0)
    assert s["intra_domain_bw_util"] == pytest.approx(0.0)


def test_the_sampler_actually_fires_once_per_arrival():
    """The hook, not the maths. A sampler wired but never called would report
    n_samples=0 and every utilisation figure would silently be absent."""
    from orion.sim.episode_runner import EpisodeRunner

    assert EpisodeRunner.on_arrival is None, "the default path must be unchanged"
    src = __import__("inspect").getsource(EpisodeRunner._handle_arrival)
    assert "self.on_arrival()" in src
    # Fired before the plan is built, so a structural reject is still sampled.
    assert src.index("self.on_arrival()") < src.index("self.plan_builder(")

    # And Plain, which does not use EpisodeRunner, samples in its own loop.
    import grid_runner as G
    plain = __import__("inspect").getsource(G.eval_plain)
    assert "cost_acc.sample_utilization()" in plain
    assert plain.index("cost_acc.sample_utilization()") < plain.index("colocation_ffd(")


def test_per_admit_cost_is_absent_rather_than_zero_when_nothing_was_admitted():
    """Dividing network cost by admissions is only meaningful with admissions.
    Reporting 0.0 would make a total-failure row look like the cheapest one."""
    sub = generate_hierarchical_topology(0)
    acc = CostAccumulator(sub)
    acc.sample_utilization()
    assert "cpu_allocated_per_admit" not in acc.utilization_summary()
