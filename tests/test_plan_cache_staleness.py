"""§AA — the plan cache replays concrete domain assignments under a key that
cannot see which domain is loaded.

This is the confound behind the "Agent B plans worse than the heuristic" reading.
`RL-alone` and `MDO-partial` call their builder FRESH on every
arrival (`eval_nonmemory` sets `pb = partial_obs_builder`, no cache wrapper), while
`Memory-off` and `Full` serve ~89-95% of arrivals from `_cached_plan_builder`.

Measured LLM-free, `partial_obs_builder` behind that same cache, seeds 42-44,
conventional, i100, 2000 arrivals:

    variant            L1      L2      L3      L4     actor_infeasible L3
    fresh            .8195   .7338   .6522   .5915          111
    cached           .8238   .6727   .5367   .4645          489
    cached+repair    .8238   .7322   .6462   .5752          114

So the cache alone costs 6.1 / 11.6 / 12.7 pp at L2/L3/L4 and is most of the gap
that was being attributed to the planner.

These tests pin the two properties that make that happen, so the mechanism cannot
quietly change without the explanation going stale with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import orion.llm.plan_cache as PC  # noqa: E402
from orion.llm.condition_signature import compute_condition_signature  # noqa: E402


@pytest.fixture(scope="module")
def substrate():
    import grid_runner as G
    G._wire("conventional", "L3", 100)
    return G._substrate_fn(100)(42)


def _drain_domain(sub, domain_id):
    """Exhaust one domain's CPU/RAM in place; return an undo callable."""
    g = sub.graph
    saved = {}
    for n in sub.nodes_in_domain(domain_id):
        saved[n] = (g.nodes[n]["cpu_residual"], g.nodes[n]["ram_residual"])
        g.nodes[n]["cpu_residual"] = 0.0
        g.nodes[n]["ram_residual"] = 0.0

    def undo():
        for nid, (c, r) in saved.items():
            g.nodes[nid]["cpu_residual"] = c
            g.nodes[nid]["ram_residual"] = r
    return undo


def test_condition_key_cannot_tell_which_domain_is_exhausted(substrate):
    """Draining domain A and draining domain B give the SAME cache key.

    The key carries `cpu_residual_frac` and per-tier totals, both summed over every
    domain. The cached plan names a specific domain. So a plan authored while
    domain B was free is served while domain B is exhausted.
    """
    assert PC.USE_PER_DOMAIN_CONDITION_KEY is False, (
        "default must stay off or every banked cell's hit rate moves")

    doms = [d for d in range(substrate.num_domains)
            if substrate.nodes_in_domain(d)][:2]
    if len(doms) < 2:
        pytest.skip("need two non-empty domains")

    keys = []
    for d in doms:
        undo = _drain_domain(substrate, d)
        try:
            keys.append(PC.condition_key(compute_condition_signature(substrate, "L3")))
        finally:
            undo()

    assert keys[0] == keys[1], (
        "condition_key already separates these; if this now passes by discriminating "
        "domains, the staleness explanation needs revisiting")


def test_per_domain_key_does_separate_them(substrate):
    """The opt-in key distinguishes them, which is what says the information is
    available and merely discarded -- `compute_condition_signature` computes
    `domain_cpu_residual` and the shipped `condition_key` drops it."""
    doms = [d for d in range(substrate.num_domains)
            if substrate.nodes_in_domain(d)][:2]
    if len(doms) < 2:
        pytest.skip("need two non-empty domains")

    PC.USE_PER_DOMAIN_CONDITION_KEY = True
    try:
        keys = []
        for d in doms:
            undo = _drain_domain(substrate, d)
            try:
                keys.append(
                    PC.condition_key(compute_condition_signature(substrate, "L3")))
            finally:
                undo()
        assert keys[0] != keys[1]
    finally:
        PC.USE_PER_DOMAIN_CONDITION_KEY = False


def test_revalidate_plan_is_topology_only_not_capacity(substrate):
    """`revalidate_plan` passes a plan whose every named domain is exhausted.

    Its docstring is explicit that residual checks are left downstream, but the
    consequence is not: downstream is the domain actor, so a stale hit becomes an
    `actor_infeasible` rejection rather than a cache miss and a fresh plan.
    """
    from partial_obs_prior import partial_obs_builder
    import grid_runner as G
    import wp7_runner as W
    import numpy as np
    from orion.sim.arrival_process import EventType

    rng = np.random.default_rng(42 + 777)
    ap = W._make_ap(substrate, 40, rng)
    ap.generate()
    sr = next(ev.slice_request for ev in ap.events
              if ev.event_type == EventType.ARRIVAL and ev.slice_request is not None)

    plan = partial_obs_builder(sr, substrate)
    assert plan is not None
    assert PC.revalidate_plan(plan, substrate) is True

    undos = [_drain_domain(substrate, d) for d in set(plan.suggested_domains)]
    try:
        assert PC.revalidate_plan(plan, substrate) is True, (
            "revalidate_plan now rejects an exhausted domain; if this was fixed "
            "deliberately, the repair-outside-the-cache layer may be redundant")
    finally:
        for u in undos:
            u()


def test_repair_sits_outside_the_cache_in_both_eval_paths():
    """The repair must wrap the CACHED builder, not the inner one.

    Wrapping the inner builder would repair only the ~11-16% of arrivals that miss
    and leave every stale hit uncorrected, which is the whole failure being fixed.
    Pinned by source order because the ordering is the fix.
    """
    import inspect
    import grid_runner as G

    for fn in (G.eval_nonmemory, G.eval_memory):
        src = inspect.getsource(fn)
        cache_at = src.index("_with_plan_cache(pb)")
        repair_at = src.index("plan_repaired(pb")
        assert cache_at < repair_at, (
            f"{fn.__name__}: plan_repaired must be applied AFTER _with_plan_cache, "
            "or stale cache hits are never revalidated against the substrate")
