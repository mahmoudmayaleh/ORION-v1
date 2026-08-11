"""§Y.5 acceptance metrics and the §Y per-cell timeout.

Pins the two properties that make these safe to rely on:
  * the rejection breakdown CONSERVES arrivals, so an unpopulated bin cannot be
    mistaken for a constraint that never binds,
  * the pre-§Y ceiling enumerator can no longer hang or run at §Y scale.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from orion.sim.acceptance import (  # noqa: E402
    REJECT_BINS,
    AcceptanceReport,
    build_report,
)


# ── doubles matching the shapes build_report reads ──────────────────────────

@dataclass
class _Violation:
    c5b_violated: bool = False
    c7_violated: bool = False
    c9_violated: bool = False
    actor_infeasible: bool = False
    cross_domain_infeasible: bool = False


@dataclass
class _Decision:
    violation: _Violation | None = None


@dataclass
class _Result:
    admitted: bool
    decision: _Decision | None = None
    revoked_by: list[str] | None = None


@dataclass
class _Stats:
    total_arrivals: int = 0
    admitted: int = 0
    rejected_structural: int = 0


@dataclass
class _Episode:
    stats: _Stats
    mdo_results: list = field(default_factory=list)


def _episode(admitted, structural, violations):
    results = [_Result(True) for _ in range(admitted)]
    results += [_Result(False, _Decision(v)) for v in violations]
    total = admitted + structural + len(violations)
    return _Episode(_Stats(total, admitted, structural), results)


# ── acceptance ──────────────────────────────────────────────────────────────

def test_acceptance_is_admitted_over_offered_not_over_a_ceiling():
    ep = _episode(admitted=30, structural=10, violations=[_Violation(c7_violated=True)] * 10)
    rep = build_report(ep)
    assert rep.offered == 50
    assert rep.acceptance == pytest.approx(0.6)


def test_breakdown_conserves_arrivals():
    """The guard that stops an unpopulated bin reading as a constraint that never binds."""
    ep = _episode(
        admitted=5,
        structural=2,
        violations=[
            _Violation(actor_infeasible=True),
            _Violation(c5b_violated=True),
            _Violation(c7_violated=True),
            _Violation(c9_violated=True),
            _Violation(cross_domain_infeasible=True),
        ],
    )
    rep = build_report(ep)
    rep.check_conservation()
    assert rep.rejections["actor_infeasible"] == 1
    assert rep.rejections["c5b_bandwidth"] == 1
    assert rep.rejections["c7_delay"] == 1
    assert rep.rejections["c9_hops"] == 1
    assert rep.rejections["cross_domain_infeasible"] == 1
    assert rep.rejections["structural"] == 2


def test_non_conserving_breakdown_raises():
    rep = AcceptanceReport(admitted=3, offered=10, rejections={"c7_delay": 2})
    with pytest.raises(ValueError, match="does not conserve"):
        rep.check_conservation()


def test_a_rejection_is_counted_once_under_precedence():
    """Multiple constraints can trip together; the columns must still sum."""
    ep = _episode(admitted=0, structural=0,
                  violations=[_Violation(actor_infeasible=True, c7_violated=True,
                                         c9_violated=True)])
    rep = build_report(ep)
    rep.check_conservation()
    assert rep.rejections["actor_infeasible"] == 1
    assert rep.rejections["c7_delay"] == 0


def test_rejection_with_no_violation_is_visible_not_dropped():
    ep = _episode(admitted=0, structural=0, violations=[None])
    rep = build_report(ep)
    rep.check_conservation()
    assert rep.rejections["unattributed"] == 1


def test_report_always_emits_every_bin():
    """A missing key and a zero must not look the same to a downstream reader."""
    rep = build_report(_episode(1, 0, [])).to_dict()
    assert set(rep["rejections"]) == set(REJECT_BINS)


def test_zero_arrivals_does_not_divide_by_zero():
    assert build_report(_episode(0, 0, [])).acceptance == 0.0


# ── post-commit revocations (§Y.13) ─────────────────────────────────────────
#
# These arrivals were COMMITTED by the coordinator, allocated, and then revoked
# by the ground-truth verifier. `decision.violation` is None on that path, so
# before the split every one of them was counted as `unattributed` (24 to 89
# percent of all rejections) and the visible c7_delay column showed only the
# pre-commit refusals, which is the opposite of where the losses actually were.

def _revoked_episode(codes_per_arrival):
    results = [_Result(False, _Decision(None), list(codes)) for codes in codes_per_arrival]
    return _Episode(_Stats(len(results), 0, 0), results)


@pytest.mark.parametrize("code,expected_bin", [
    ("C2", "post_commit_c2_cpu"),
    ("C3", "post_commit_c3_ram"),
    ("C5b", "post_commit_c5b_bandwidth"),
    ("C7", "post_commit_c7_delay"),
    ("PLAN_BUILD", "post_commit_plan_build"),
])
def test_post_commit_revocation_is_attributed_not_dropped(code, expected_bin):
    rep = build_report(_revoked_episode([[code]]))
    rep.check_conservation()
    assert rep.rejections[expected_bin] == 1
    assert rep.rejections["unattributed"] == 0, (
        f"{code} revocation fell back into unattributed")


def test_post_commit_beats_the_empty_pre_commit_record():
    """The exact shape that produced the bug: committed, so no pre-commit
    violation exists, then revoked under load."""
    ep = _revoked_episode([["C7"]])
    assert ep.mdo_results[0].decision.violation is None
    assert build_report(ep).rejections["post_commit_c7_delay"] == 1


def test_post_commit_precedence_counts_each_revocation_once():
    rep = build_report(_revoked_episode([["C7", "C2", "C9"]]))
    rep.check_conservation()
    assert rep.rejections["post_commit_c2_cpu"] == 1
    assert rep.rejections["post_commit_c7_delay"] == 0


def test_pre_and_post_commit_c7_are_separate_columns():
    """The claim the taxonomy exists to support: a delay rejection the
    coordinator made and one the verifier made are different events."""
    pre = _Result(False, _Decision(_Violation(c7_violated=True)))
    post = _Result(False, _Decision(None), ["C7"])
    rep = build_report(_Episode(_Stats(2, 0, 0), [pre, post]))
    rep.check_conservation()
    assert rep.rejections["c7_delay"] == 1
    assert rep.rejections["post_commit_c7_delay"] == 1


def test_unrecognised_revocation_code_stays_visible():
    """A future verifier code must not be silently dropped from the sum."""
    rep = build_report(_revoked_episode([["C99"]]))
    rep.check_conservation()
    assert rep.rejections["unattributed"] == 1


# ── the commit that cannot be realised (§Y.13) ──────────────────────────────

def test_unrealisable_commit_is_rejected_not_counted_as_admitted():
    """Regression, `episode_runner._handle_arrival`: when the coordinator commits
    a partition but no concrete placement can be built, nothing is allocated and
    nothing is tracked. Leaving `admitted` True counted a slice that does not
    exist on the substrate, inflating acceptance, and its departure was a no-op.

    Checked on the source because the surrounding branch needs a live substrate,
    a coordinator and an arrival stream to reach; the property is that the branch
    exists and revokes.
    """
    import ast
    import inspect

    from orion.sim import episode_runner

    tree = ast.parse(inspect.getsource(episode_runner))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "placement_plan is None" not in test:
            continue
        body = "\n".join(ast.unparse(s) for s in node.body)
        found.append(body)

    assert found, "the placement_plan-is-None branch is gone; the silent path is back"
    assert any("admitted = False" in b for b in found), (
        "an unrealisable commit no longer revokes admission")
    assert any("PLAN_BUILD" in b for b in found), (
        "the revocation records no reason, so it will bin as unattributed")


def test_episode_stats_counts_unrealisable_commits():
    from orion.sim.episode_runner import EpisodeStats

    assert EpisodeStats().plan_build_failures == 0


# ── the QoS gate reports which constraint it refused on (§Y.13) ─────────────

def _gate_fixture(domains, hop_budget_flow, delay_budget):
    """Two nodes, one link, one VNF routed across it."""
    import types

    import networkx as nx

    g = nx.Graph()
    for nid, dom in zip(("n0", "n1"), domains):
        g.add_node(nid, domain_id=dom, cpu_capacity=100.0, cpu_residual=90.0,
                   processing_delay=0.001)
    g.add_edge("n0", "n1", link_id="l0", bandwidth_capacity=100.0,
               bw_residual=90.0, propagation_delay=0.001)

    substrate = types.SimpleNamespace(graph=g)
    plan = types.SimpleNamespace(
        vnf_placements={"v0": "n0"},
        flow_routes={("v0", "v1"): ["l0"] if hop_budget_flow else []},
        bw_allocations={("v0", "v1"): {"l0": 1.0}} if hop_budget_flow else {},
    )
    vnf = types.SimpleNamespace(vnf_id="v0", cpu_demand=1.0,
                                computational_intensity=1.0)
    slice_req = types.SimpleNamespace(
        vnfs=[vnf], qos=types.SimpleNamespace(max_e2e_delay=delay_budget))
    return substrate, slice_req, plan


def test_qos_gate_names_the_hop_constraint():
    from orion.sim.qos_gate import plan_qos_reason

    sub, sr, plan = _gate_fixture(("d0", "d1"), hop_budget_flow=True, delay_budget=1e9)
    assert plan_qos_reason(sub, sr, plan, max_inter_domain_hops=0) == "C9"


def test_qos_gate_names_the_delay_constraint():
    from orion.sim.qos_gate import plan_qos_reason

    sub, sr, plan = _gate_fixture(("d0", "d0"), hop_budget_flow=True, delay_budget=1e-9)
    assert plan_qos_reason(sub, sr, plan, max_inter_domain_hops=8) == "C7"


def test_qos_gate_passes_a_comfortable_plan():
    from orion.sim.qos_gate import plan_qos_reason

    sub, sr, plan = _gate_fixture(("d0", "d0"), hop_budget_flow=True, delay_budget=1e9)
    assert plan_qos_reason(sub, sr, plan, max_inter_domain_hops=8) is None


@pytest.mark.parametrize("domains,hops,budget", [
    (("d0", "d1"), 0, 1e9),      # C9
    (("d0", "d0"), 8, 1e-9),     # C7
    (("d0", "d0"), 8, 1e9),      # pass
])
def test_qos_ok_agrees_with_qos_reason(domains, hops, budget):
    """The gate runs on the coordinator's commit path as well as Plain's, so a
    second implementation that drifted would move acceptance everywhere and be
    blamed on the taxonomy."""
    from orion.sim.qos_gate import plan_qos_ok, plan_qos_reason

    sub, sr, plan = _gate_fixture(domains, hop_budget_flow=True, delay_budget=budget)
    assert (plan_qos_reason(sub, sr, plan, hops) is None) is plan_qos_ok(sub, sr, plan, hops)


def test_plain_reports_a_rejection_taxonomy():
    """Plain is the reference approach and was the one row missing from every mix
    table, because `eval_plain` returned no `rejections` key."""
    import inspect

    import grid_runner as G

    src = inspect.getsource(G.eval_plain)
    assert '"rejections"' in src, "Plain still returns no rejection breakdown"
    assert "check_conservation" in src, "Plain's breakdown is not conservation-checked"


# ── ceiling enumerator: cannot hang, refuses §Y scale ───────────────────────

def test_ceiling_refuses_y_scale_substrates():
    """§Y Large is 100 nodes. The oracle is contention-blind AND unbounded there,
    so it must refuse loudly rather than park a core."""
    import approach_runner as F
    from orion.substrate.hierarchical_topology import generate_hierarchical_topology

    sub = generate_hierarchical_topology(0)
    with pytest.raises(RuntimeError, match="compute_ceiling refused"):
        F.compute_ceiling(sub, arrival_seed=42, num_arrivals=5)


def test_ceiling_never_materialises_the_placement_product():
    """Regression: the old code built the full product BEFORE down-sampling it, so
    the 5000 cap could not save it (40 nodes, K=5 => 1.0e8 tuples allocated)."""
    import approach_runner as F

    # Comments legitimately quote the old line to explain why it went, so only
    # executable lines are checked.
    code = "\n".join(
        line for line in Path(F.__file__).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "list(itertools.product(*feasible_nodes))" not in code
    assert "itertools.islice(itertools.product(" in code


# ── per-cell timeout ────────────────────────────────────────────────────────

def test_cell_timeout_fires_or_no_ops_cleanly():
    import signal
    import time

    import grid_runner as G

    if not hasattr(signal, "SIGALRM"):
        with G.cell_timeout(1, "probe"):
            pass  # platform without SIGALRM: must be a silent no-op, not an error
        pytest.skip("SIGALRM unavailable on this platform")

    with pytest.raises(G.CellTimeout, match="exceeded"):
        with G.cell_timeout(1, "probe"):
            time.sleep(3)


def test_timeout_is_disarmed_after_a_fast_cell():
    """A leaked alarm would fire during an unrelated later cell."""
    import signal

    import grid_runner as G

    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM unavailable on this platform")
    with G.cell_timeout(30, "fast"):
        pass
    assert signal.alarm(0) == 0, "alarm still armed after the cell returned"


# ── the load level must actually reach every stream (§Y.3) ───────────────────

def test_every_y_runner_stream_uses_the_calibrated_rate():
    """No `ArrivalProcess` in a §Y runner may be built from the module constants.

    Regression, found 2026-07-30 in a live run: `eval_plain` built its stream with
    `W.ARRIVAL_RATE` / `W.SERVICE_RATE`, the pre-§Y fixed load. Plain therefore
    returned the SAME acceptance (0.727) at L1, L2, L3 and L4 while every other
    approach varied correctly, so the load axis would have been reported against a
    flat baseline. The M^B warm-up had the same defect.

    This is the silent-no-op class: the run completes and every cell carries a
    plausible number, so nothing surfaces until the levels are compared.

    Checked across ALL THREE §Y runners, not just the one where it was found.
    Pre-§Y probe scripts are out of scope: they produce no §Y cell and several
    legitimately sweep their own rate.
    """
    import ast
    import inspect

    import grid_runner
    import milp_approach_runner
    import wp7_runner

    # module -> the ONE function allowed to fall back to the module constants,
    # which happens only when no level is wired at all (smoke runs). None = the
    # module may not mention them in a stream at all.
    ALLOWED_FALLBACK = {
        "grid_runner": "_ap_for_level",
        "wp7_runner": "_make_ap",
        "milp_approach_runner": None,
    }

    offenders = []
    for mod in (grid_runner, wp7_runner, milp_approach_runner):
        name = mod.__name__
        tree = ast.parse(inspect.getsource(mod))
        # Map each ArrivalProcess call to its enclosing function, so "allowed"
        # means allowed *there* rather than anywhere in the file.
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", None) == "ArrivalProcess"):
                    continue
                call = ast.unparse(node)
                if "ARRIVAL_RATE" not in call and "SERVICE_RATE" not in call:
                    continue
                if fn.name == ALLOWED_FALLBACK.get(name):
                    continue
                offenders.append(f"{name}.{fn.name}: {call}")

    assert not offenders, (
        "these streams bypass the calibrated load level and would make their "
        f"cells identical at every level: {offenders}")


def test_plain_eval_rate_tracks_the_wired_level():
    """Behavioural counterpart: the rate Plain's stream is built at must change when
    the wired level changes. Pins the effect, not just the source text.

    Two regimes, and the test asserts in BOTH rather than skipping in one. While the
    ladder is uncalibrated (as after the §Y.1e substrate change) the correct
    behaviour is to REFUSE, because a fallback lambda would produce a complete run
    measured at a load nobody chose. Skipping here would leave the guard silently
    inactive for exactly as long as the refusal is in force, which is when a stale
    lambda is most likely to be reintroduced.
    """
    import grid_runner as G
    from orion.sim.load_levels import CALIBRATED_LEVELS

    if not CALIBRATED_LEVELS:
        with pytest.raises(RuntimeError, match="calibration has not been run"):
            G._wire("conventional", "L1", 0)
        return

    seen = {}
    for name in ("L1", "L4"):
        G._wire("conventional", name, 0)
        ap = G._ap_for_level(object(), 10, None)
        seen[name] = ap.arrival_rate if hasattr(ap, "arrival_rate") else None

    assert seen["L1"] is not None, "could not read the arrival rate off the stream"
    assert seen["L1"] != seen["L4"], (
        f"Plain's arrival rate is identical at L1 and L4 ({seen}); the load level "
        "is not reaching the stream")
