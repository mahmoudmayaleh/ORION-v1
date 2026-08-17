"""§AA — the LLM plan path held to the heuristic's admissibility tests.

The property these pin is the one the guard exists to enforce, and it is exactly
the kind that can silently stop holding: `repair_plan` is a pass-through on almost
every code path, so a wiring mistake, an unset mode or a canonical/raw index slip
would make it a no-op that still returns a plausible plan and still banks a cell.
See the silent-no-op guard principle.

The measurement that motivates the guard (banked conventional/i100/2000-arrival
cells, 5 seeds):

    approach      L1     L2     L3     L4     cross_domain_infeasible + c9_hops
    MDO-partial  .817   .732   .594   .512     0 /   0 /  15 /  15
    RL-alone     .835   .731   .584   .511    12 /  20 /   7 /   6
    Full         .722   .658   .536   .445   236 / 220 / 183 / 146

Both of those bins are reachable ONLY through a split partition
(MDOCoordinator._build_fragments routes a flow cross-domain iff its endpoints
differ), `partial_obs_builder` splits 0.0% of arrivals, and a whole-chain host is
node-feasible on 100% of arrivals. So the Full-minus-RL-alone gap is elective
splitting by the planner, not the h^m encoding defect of
docs/OBS_ENCODING_DEFECT_2026-08-15.md.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import partial_obs_prior as POP  # noqa: E402
from orion.mdo.observation import build_domain_summaries  # noqa: E402


# ── fixtures: a real §Y substrate and a real arrival stream, no LLM ──────────────
@pytest.fixture(scope="module")
def stream():
    """(substrate, [slice_request, ...]) on the wired §Y eval instance at L3.

    L3 rather than L1 because the guard has nothing to do on an empty network:
    every domain is admissible, so `repair_plan` returns its input and a test that
    passed there would prove nothing.
    """
    import numpy as np

    import grid_runner as G
    import wp7_runner as W
    from orion.sim.arrival_process import EventType

    G._wire("conventional", "L3", 100)
    sub = G._substrate_fn(100)(42)
    rng = np.random.default_rng(42 + 777)
    ap = W._make_ap(sub, 120, rng)
    ap.generate()
    reqs = [ev.slice_request for ev in ap.events
            if ev.event_type == EventType.ARRIVAL and ev.slice_request is not None]
    assert reqs, "no arrivals generated"
    return sub, reqs


def _split_plan(sr, sub):
    """A deliberately scattered m̃ over node-feasible domains: what the planner
    authors on the arrivals that cost ORION the gap. Falls back to None when the
    request cannot be scattered, so callers can skip it."""
    summaries = build_domain_summaries(sub)
    dom_nodes = [set(sub.nodes_in_domain(s.domain_id)) for s in summaries]
    feas = [[m for m in range(len(summaries))
             if set(v.permitted_nodes) & dom_nodes[m]] for v in sr.vnfs]
    if any(not f for f in feas):
        return None
    doms, used = [], set()
    for row in feas:                      # prefer an unused domain -> maximal scatter
        pick = next((m for m in row if m not in used), row[0])
        used.add(pick)
        doms.append(summaries[pick].domain_id)
    if len(set(doms)) < 2:
        return None
    base = POP.partial_obs_builder(sr, sub)
    if base is None:
        return None
    return replace(base, suggested_domains=doms)


# ── the builder property the whole diagnosis rests on ───────────────────────────
def test_partial_obs_builder_never_splits_when_a_host_exists(stream):
    """RL-alone's and MDO-partial's plan source colocates whenever it can.

    This is the control for the Full comparison. If it ever starts splitting, the
    "elective split" reading of the gap is void and the guard is aimed wrong.
    """
    sub, reqs = stream
    summaries = build_domain_summaries(sub)
    dom_nodes = [set(sub.nodes_in_domain(s.domain_id)) for s in summaries]
    checked = 0
    for sr in reqs:
        colocatable = any(
            all(set(v.permitted_nodes) & dn for v in sr.vnfs) for dn in dom_nodes)
        if not colocatable:
            continue
        plan = POP.partial_obs_builder(sr, sub)
        if plan is None:
            continue
        checked += 1
        assert len(set(plan.suggested_domains)) == 1, (
            f"partial_obs_builder split a colocatable chain: "
            f"{plan.suggested_domains}")
    assert checked > 0, "no colocatable arrival was exercised"


# ── the guard must not move a single shipped cell ───────────────────────────────
def test_repair_off_is_identity(stream):
    """`off` is the default and is what every banked cell was produced with."""
    assert POP.REPAIR_MODE == "off", (
        "ORION_PLAN_REPAIR must default to 'off' or the shipped cells stop "
        "reproducing")
    sub, reqs = stream
    for sr in reqs[:30]:
        plan = POP.partial_obs_builder(sr, sub)
        assert POP.repair_plan(plan, sr, sub, mode="off") is plan


def test_repair_rejects_unknown_mode(stream):
    sub, reqs = stream
    plan = POP.partial_obs_builder(reqs[0], sub)
    with pytest.raises(ValueError, match="unknown ORION_PLAN_REPAIR mode"):
        POP.repair_plan(plan, reqs[0], sub, mode="collapse")


# ── what the guard actually does ────────────────────────────────────────────────
def test_full_mode_collapses_an_elective_split(stream):
    """The headline behaviour: a scattered m̃ becomes a single-domain m̃ whenever a
    whole-chain host exists. Without this the guard is a no-op on exactly the
    arrivals that cost ORION the gap."""
    sub, reqs = stream
    POP.reset_repair_stats()
    collapsed = attempted = 0
    for sr in reqs:
        split = _split_plan(sr, sub)
        if split is None:
            continue
        view = POP._PartitionView(sr, sub)
        host, _ = view.colocation_candidate()
        if host is None:
            continue
        attempted += 1
        out = POP.repair_plan(split, sr, sub, mode="full")
        assert len(set(out.suggested_domains)) == 1, (
            f"split survived the full guard: {out.suggested_domains}")
        assert out.suggested_domains[0] == view.summaries[host].domain_id
        collapsed += 1
    assert attempted > 0, "no splittable arrival with a colocation host was found"
    assert collapsed == attempted
    assert POP.REPAIR_STATS["collapsed"] == collapsed


def test_guard_mode_preserves_the_split(stream):
    """`guard` isolates h^m: it repairs inadmissible assignments and nothing else,
    so a split the planner authored survives. This is what makes the
    Memory-off-rpg / Memory-off-rp pair a decomposition rather than one number."""
    sub, reqs = stream
    seen = 0
    for sr in reqs:
        split = _split_plan(sr, sub)
        if split is None:
            continue
        view = POP._PartitionView(sr, sub)
        if view.colocation_candidate()[0] is None:
            continue
        out = POP.repair_plan(split, sr, sub, mode="guard")
        if all(view.admissible(k, view.domain_to_canonical[d])
               for k, d in enumerate(split.suggested_domains)):
            assert out.suggested_domains == split.suggested_domains, (
                "guard mode moved an admissible assignment")
            seen += 1
    assert seen > 0, "no admissible split was exercised"


def test_repair_output_is_always_node_feasible(stream):
    """Every domain the guard chooses holds a permitted node for its VNF, i.e. it
    stays inside the coordinator's hard mask. A violation here fires the
    node-based frame assert at COMMIT rather than showing up as a lost point."""
    sub, reqs = stream
    dom_nodes = {d: set(sub.nodes_in_domain(d)) for d in range(sub.num_domains)}
    for mode in ("guard", "full"):
        for sr in reqs:
            split = _split_plan(sr, sub)
            if split is None:
                continue
            out = POP.repair_plan(split, sr, sub, mode=mode)
            for k, d in enumerate(out.suggested_domains):
                assert set(sr.vnfs[k].permitted_nodes) & dom_nodes[d], (
                    f"{mode}: VNF {k} sent to domain {d}, which holds none of its "
                    f"permitted nodes")


def test_repair_never_turns_a_plan_into_a_structural_reject(stream):
    """A non-None plan in must be a non-None plan out, in every mode.

    Returning None would move the arrival into the `structural` rejection bin, and
    the guard would then be scored on rejections it relabelled rather than on
    rejections it converted.
    """
    sub, reqs = stream
    for mode in ("off", "guard", "full"):
        for sr in reqs:
            for plan in (POP.partial_obs_builder(sr, sub), _split_plan(sr, sub)):
                if plan is None:
                    continue
                assert POP.repair_plan(plan, sr, sub, mode=mode) is not None


def test_repair_preserves_every_field_but_the_partition(stream):
    """The guard reassigns domains. It must not touch demands, vcrs, bw or the VNF
    identities, or it stops being the same request."""
    sub, reqs = stream
    for sr in reqs[:40]:
        split = _split_plan(sr, sub)
        if split is None:
            continue
        out = POP.repair_plan(split, sr, sub, mode="full")
        assert out.vnf_ids == split.vnf_ids
        assert out.required_tiers == split.required_tiers
        assert out.cpu_demands == split.cpu_demands
        assert out.ram_demands == split.ram_demands
        assert out.vcrs == split.vcrs
        assert out.bw_demands == split.bw_demands
        assert len(out.suggested_domains) == len(split.suggested_domains)


def test_repair_is_idempotent(stream):
    """Repairing a repaired plan changes nothing: the guard's output is admissible
    by its own test, which is what lets it sit in front of the plan cache."""
    sub, reqs = stream
    for mode in ("guard", "full"):
        for sr in reqs[:40]:
            split = _split_plan(sr, sub)
            if split is None:
                continue
            once = POP.repair_plan(split, sr, sub, mode=mode)
            twice = POP.repair_plan(once, sr, sub, mode=mode)
            assert twice.suggested_domains == once.suggested_domains


# ── wiring: the runner must be able to ask for these, and only these ────────────
def test_runner_exposes_the_repair_approaches_without_touching_the_base_rows():
    import grid_runner as G

    for ap, mode in G.REPAIR_APPROACHES.items():
        assert ap in G.APPROACHES
        assert mode in ("guard", "full")
        base = G.REPAIR_BASE[ap]
        assert base in G.APPROACHES
        # A repair cell must run the same weights as the row it is compared to.
        assert G.STACK_FOR_APPROACH[ap] == G.STACK_FOR_APPROACH[base]
        assert (ap in G.ADVISED_APPROACHES) == (base in G.ADVISED_APPROACHES)
        assert G.MEMORY_APPROACHES.get(ap) == G.MEMORY_APPROACHES.get(base)
        assert ap in G.TRAINED_APPROACHES
    # The base rows must not have acquired a repair by being listed here.
    assert set(G.REPAIR_APPROACHES) & set(G.REPAIR_BASE.values()) == set()
