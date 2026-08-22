"""§AF — chain order is a CONSTRAINT (C10), enforced on every approach.

Two changes land together here (2026-08-21, user directive):

  * `partial_obs_builder` no longer searches for a single domain that can hold the
    whole chain. That branch decided 100% of banked arrivals, so the heuristic
    answered every partitioning question with "one domain" and the multi-domain
    problem was never posed. What is left is per-VNF best-fit.
  * The partition must be chain-contiguous: each domain occupies exactly one
    maximal run. `MDOCoordinator` refuses a partition that re-enters a domain it
    had left, before dispatch, binned as `chain_order`; `orion.mdo.chain_order` is
    the single definition and `Plain` is held to it too.

These are the properties that can silently stop holding: a flag read at import, a
filter that never removes anything, an enforcement point that a second code path
walks around. Each is pinned rather than described.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import partial_obs_prior as POP  # noqa: E402
from orion.mdo import chain_order  # noqa: E402


@pytest.fixture(scope="module")
def stream():
    """(substrate, [slice_request, ...]) on the wired §Y eval instance at L3."""
    import numpy as np

    import grid_runner as G
    import wp7_runner as W
    from orion.sim.arrival_process import EventType

    G._wire("conventional", "L3", 100)
    sub = G._substrate_fn(100)(42)
    rng = np.random.default_rng(42 + 777)
    ap = W._make_ap(sub, 200, rng)
    ap.generate()
    reqs = [ev.slice_request for ev in ap.events
            if ev.event_type == EventType.ARRIVAL and ev.slice_request is not None]
    assert reqs, "no arrivals generated"
    return sub, reqs


# ── the rule itself ─────────────────────────────────────────────────────────────
def test_is_contiguous_cases():
    assert chain_order.is_contiguous([])
    assert chain_order.is_contiguous([1, 1, 1])
    assert chain_order.is_contiguous([1, 1, 2])
    assert chain_order.is_contiguous([1, 2, 2, 3])
    assert not chain_order.is_contiguous([1, 2, 1])
    assert not chain_order.is_contiguous([1, 2, 3, 2])


def test_run_count():
    assert chain_order.run_count([1, 1, 2, 2, 1]) == 3
    assert chain_order.run_count([4]) == 1


def test_defaults_are_on(monkeypatch):
    """Both halves default ON. The whole point is that they are not opt-in."""
    monkeypatch.delenv("ORION_CHAIN_ORDER", raising=False)
    assert chain_order.enabled(), "C10 must default on"
    assert POP.CONTIGUOUS is True, "ORION_PARTIAL_CONTIG must default on"


def test_off_switch_is_honoured(monkeypatch):
    """The ablation that measures what C10 costs must still be reachable."""
    monkeypatch.setenv("ORION_CHAIN_ORDER", "off")
    assert not chain_order.enabled()
    assert not chain_order.violates([1, 2, 1])


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv("ORION_CHAIN_ORDER", "repair")
    with pytest.raises(ValueError, match="unknown ORION_CHAIN_ORDER"):
        chain_order.enabled()


# ── the builder ────────────────────────────────────────────────────────────────
def test_builder_no_longer_colocates_by_rule(stream):
    """The colocation-first branch is GONE, not disabled.

    Pinning its absence rather than its effect: an effect test would pass again
    the moment someone reintroduced the branch behind a default-on flag.
    """
    import inspect
    src = inspect.getsource(POP.partial_obs_builder)
    assert "colocation_candidate()" not in src, (
        "partial_obs_builder is choosing a whole-chain host again")


def test_builder_output_is_contiguous(stream):
    sub, reqs = stream
    checked = 0
    for sr in reqs:
        plan = POP.partial_obs_builder(sr, sub)
        if plan is None:
            continue
        checked += 1
        assert chain_order.is_contiguous(plan.suggested_domains), (
            f"builder authored a partition C10 would refuse: "
            f"{plan.suggested_domains}")
    assert checked > 0, "no arrival produced a plan"


def test_builder_places_every_vnf(stream):
    sub, reqs = stream
    for sr in reqs:
        plan = POP.partial_obs_builder(sr, sub)
        if plan is None:
            continue
        assert len(plan.suggested_domains) == len(sr.vnfs)


def test_unconstrained_builder_would_violate(stream, monkeypatch):
    """C10 is not vacuous: with the restriction off, the builder does re-enter.

    Without this the contiguity test above could pass because nothing ever splits.
    """
    sub, reqs = stream
    monkeypatch.setattr(POP, "CONTIGUOUS", False)
    bad = sum(1 for sr in reqs
              for p in [POP.partial_obs_builder(sr, sub)]
              if p is not None and not chain_order.is_contiguous(p.suggested_domains))
    assert bad > 0, ("the unconstrained builder produced no non-contiguous "
                     "partition; this suite no longer proves anything")


# ── enforcement reaches the coordinator ────────────────────────────────────────
def test_violation_is_carried_and_binned():
    from orion.mdo.types import ViolationInfo
    from orion.sim.acceptance import REJECT_BINS, _classify

    v = ViolationInfo(chain_order_violated=True)
    assert v.has_violation, "C10 must make the decision fail"
    assert "chain_order" in REJECT_BINS

    class _D:
        violation = v

    class _R:
        decision = _D()
        revoked_by = None

    assert _classify(_R()) == "chain_order"


def test_coordinator_enforces_before_dispatch():
    """The check must sit above the actors, not inside one of them."""
    import inspect

    from orion.mdo.coordinator import MDOCoordinator

    src = inspect.getsource(MDOCoordinator.resolve_arrival)
    guard = src.index("chain_order.violates(partition)")
    dispatch = src.index("_dispatch_to_actors")
    assert guard < dispatch, "C10 is evaluated after the actors have run"
