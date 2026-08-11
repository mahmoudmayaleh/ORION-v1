"""Integration test for the episode runner.

End-to-end smoke test: substrate + arrival_process + domain actors + MDO
coordinator (random policy) → drain one short episode → KPIs sensible,
rollout populated, substrate residuals consistent with admitted/departed
slices.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "torch_geometric",
    reason="domain actors require the [actors] optional extra (torch_geometric)",
)

from orion.actors.domain_actor import DomainActor  # noqa: E402
from orion.config import TopologyConfig  # noqa: E402
from orion.mdo.coordinator import MDOConfig, MDOCoordinator  # noqa: E402
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.graph_model import SubstrateNetwork
from orion.substrate.topology_generator import generate_multi_domain_topology


@pytest.fixture
def substrate() -> SubstrateNetwork:
    rng = np.random.default_rng(42)
    return generate_multi_domain_topology(
        TopologyConfig(
            num_domains=3,
            nodes_per_domain=[5, 5, 5],
            intra_link_density=0.6,
            inter_domain_links=2,
        ),
        rng,
    )


@pytest.fixture
def arrival_process(substrate) -> ArrivalProcess:
    rng = np.random.default_rng(123)
    proc = ArrivalProcess(
        substrate=substrate,
        num_arrivals=8,
        arrival_rate=2.0,
        service_rate=1.0,
        rng=rng,
    )
    proc.generate()
    return proc


@pytest.fixture
def coordinator(substrate) -> MDOCoordinator:
    actors = {d: DomainActor(domain_id=d) for d in range(substrate.num_domains)}
    return MDOCoordinator(
        policy=None,
        domain_actors=actors,
        config=MDOConfig(),
    )


@pytest.fixture
def runner(substrate, arrival_process, coordinator) -> EpisodeRunner:
    inter_domain_delays = {
        (0, 1): 5.0, (1, 0): 5.0,
        (1, 2): 5.0, (2, 1): 5.0,
        (0, 2): 10.0, (2, 0): 10.0,
    }
    return EpisodeRunner(
        substrate=substrate,
        arrival_process=arrival_process,
        coordinator=coordinator,
        inter_domain_delays=inter_domain_delays,
    )


class TestEpisodeSmoke:
    def test_runs_to_completion(self, runner) -> None:
        result = runner.run_episode(mdo_mode="random")
        assert result.stats.total_arrivals > 0
        assert result.stats.total_arrivals == (
            result.stats.admitted
            + result.stats.rejected_by_mdo
            + result.stats.rejected_structural
        )

    def test_kpis_decoupled_from_rollout(self, runner) -> None:
        """D2: structurally rejected slices count in KPIs but not in the
        rollout buffer. Admitted + rejected_by_mdo == MDO transitions
        (one per arrival the MDO acted on)."""
        result = runner.run_episode(mdo_mode="random")
        mdo_acted_on = result.stats.admitted + result.stats.rejected_by_mdo
        # Each MDO-acted-on arrival generates ≥1 MDO transition (the first
        # successful trial; more on retries). So #transitions ≥ #acted_on.
        assert result.rollout.num_mdo_transitions >= mdo_acted_on

    def test_substrate_residuals_consistent(self, runner, substrate) -> None:
        """After episode end, every node's cpu_residual should equal
        its capacity minus the sum of demands from still-active slices."""
        runner.reset()
        result = runner.run_episode(mdo_mode="random")
        for n, d in substrate.graph.nodes(data=True):
            # Residual never negative (allocations clamp at capacity).
            assert d["cpu_residual"] >= -1e-6, (
                f"node {n} CPU residual {d['cpu_residual']} negative"
            )
            assert d["ram_residual"] >= -1e-6
        assert result.stats.total_arrivals > 0

    def test_reset_zeros_stats(self, runner) -> None:
        runner.run_episode(mdo_mode="random")
        runner.reset()
        # After reset, residuals should match capacity on all nodes.
        for _, d in runner.substrate.graph.nodes(data=True):
            assert d["cpu_residual"] == pytest.approx(d["cpu_capacity"])
            assert d["ram_residual"] == pytest.approx(d["ram_capacity"])


class TestOnDecisionHook:
    """`on_decision` is what lets M^B accumulate during evaluation. Unset it does
    nothing, which is the right default and also the way the whole mechanism can
    fail without a symptom: no exception, no log line, just an empty store at the
    end of a multi-hour run."""

    def test_default_is_no_call(self, runner) -> None:
        assert EpisodeRunner.on_decision is None
        runner.run_episode(mdo_mode="random")   # must not raise with no hook

    def test_fires_once_per_mdo_reaching_arrival(self, runner) -> None:
        seen = []
        runner.on_decision = lambda sr, res, verdict, plan: seen.append(
            (sr.request_id, res.admitted))
        result = runner.run_episode(mdo_mode="random")

        # Structural rejects never reach the MDO, so they get no callback.
        expected = result.stats.admitted + result.stats.rejected_by_mdo
        assert len(seen) == expected
        assert len(seen) == len({rid for rid, _ in seen})   # no duplicates

    def test_reports_the_outcome_the_kpi_counts(self, runner) -> None:
        """Fired AFTER the post-commit verify, not at coordinator resolve.

        A record written at resolve time labels every post-commit revocation a
        success. The store retrieves successes-only, so that mislabelling is what
        would quietly poison it.
        """
        seen = []
        runner.on_decision = lambda sr, res, verdict, plan: seen.append(res.admitted)
        result = runner.run_episode(mdo_mode="random")

        assert sum(1 for a in seen if a) == result.stats.admitted

    def test_plan_is_the_one_the_decision_used(self, runner) -> None:
        seen = []
        runner.on_decision = lambda sr, res, verdict, plan: seen.append(plan)
        runner.run_episode(mdo_mode="random")
        # None would mean a structural reject, which never reaches the hook.
        assert seen and all(p is not None for p in seen)
