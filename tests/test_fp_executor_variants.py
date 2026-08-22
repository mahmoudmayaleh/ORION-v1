"""§AB — the executor is a separate term from the plan, and "-fp" isolates it.

`MDO-partial` runs mode="follow_prior" (the partition is committed as authored, the
policy is never consulted). `Memory-off`/`Full` run mode="advised" (the trained
policy samples and the plan only biases it). So the two published rows differ in
the PLAN and in the EXECUTOR simultaneously.

Measured 2026-08-19 on L3, seed 42, 30 arrivals: `repair_plan(mode="full")`
reproduces `partial_obs_builder`'s partition on 83% of arrivals with identical
required_tiers on 100%, yet scored .3795 through "advised" against `MDO-partial`'s
.6522 through "follow_prior". These tests pin the wiring that lets that be
measured instead of inferred.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import grid_runner as G  # noqa: E402


def test_fp_approaches_are_fully_registered():
    """Every derived table must know an "-fp" variant, or it dispatches with no
    stack / builds no Agent B -- the exact failure `Memory-off-rp` already hit."""
    assert G.FP_APPROACHES, "no -fp variants declared"
    for a in G.FP_APPROACHES:
        base = G.FP_BASE[a]
        assert a in G.APPROACHES, f"{a} not selectable on the command line"
        assert a in G.TRAINED_APPROACHES, f"{a} would dispatch without a stack"
        assert G.STACK_FOR_APPROACH[a] == G.STACK_FOR_APPROACH[base]
        assert a in G.LLM_APPROACHES, f"{a} is LLM-planned but builds no agent"


def test_fp_runs_follow_prior_not_advised():
    """The whole point: an -fp approach must commit the plan verbatim.

    Pinned on the source of `_mdo_mode` because the precedence is the fix -- every
    -fp variant is also in ADVISED_APPROACHES by inheritance, so an `advised`-first
    ordering would silently give it the wrong executor.
    """
    src = inspect.getsource(G._mdo_mode)
    assert "FP_APPROACHES" in src, "_mdo_mode does not consult FP_APPROACHES"
    fp_at = src.index("FP_APPROACHES")
    adv_at = src.index("ADVISED_APPROACHES")
    assert fp_at < adv_at, (
        "follow_prior must be tested BEFORE advised, or -fp approaches inherit "
        "mode='advised' and measure nothing")


def test_both_dispatch_paths_derive_the_mode_from_one_place():
    """§AE.1 -- `eval_memory` hardcoded mode="advised" while `eval_nonmemory`
    derived it, so `Full-rpc-fp` would have run the trained decode under a name that
    says the plan is committed verbatim. Two definitions is how it drifted; this
    pins that there is now one.
    """
    for fn in (G.eval_nonmemory, G.eval_memory):
        src = inspect.getsource(fn)
        assert "_mdo_mode(approach)" in src, (
            f"{fn.__name__} does not derive its decode mode from _mdo_mode")
        assert 'mode="advised"' not in src, (
            f"{fn.__name__} still hardcodes a decode mode")
    assert "approach" in inspect.signature(G.eval_memory).parameters, (
        "eval_memory cannot derive a mode it is never told the approach for")


@pytest.mark.parametrize("approach,expected", [
    ("Memory-off-fp", "follow_prior"),
    ("Memory-off-rpg-fp", "follow_prior"),
    ("Memory-off-rpc-fp", "follow_prior"),
    ("Full-rpc-fp", "follow_prior"),
    ("Memory-off", "advised"),
    ("Memory-off-rpc", "advised"),
    ("Full-rpc", "advised"),
    ("RL-alone", "deterministic"),
])
def test_mode_for_approach(approach, expected):
    assert G._mdo_mode(approach) == expected


def test_full_rpc_variants_are_fully_registered():
    """§AE.1 -- the contract + colo repair on the approach the paper reports.

    Every §AA-§AE result is `Memory-off`, which the 2026-08-13 directive removed
    from the paper, so these two are what make the work citable. They must keep
    M^B on: a Full variant that silently dropped the memory would be Memory-off
    under another name.
    """
    for a in ("Full-rpc", "Full-rpc-fp"):
        assert a in G.APPROACHES, f"{a} not selectable on the command line"
        assert G._variant_base(a) == "Full"
        assert a in G.MEMORY_LLM_APPROACHES, f"{a} would dispatch to the non-M^B path"
        assert G.MEMORY_APPROACHES.get(a) == "selective", f"{a} lost its M^B write policy"
        assert G.REPAIR_APPROACHES.get(a) == "colo"
        assert G.STACK_FOR_APPROACH[a] == G.STACK_FOR_APPROACH["Full"]
        assert a in G.LLM_APPROACHES, f"{a} is LLM-planned but builds no agent"
    assert "Full-rpc-fp" in G.FP_APPROACHES
    assert "Full-rpc" not in G.FP_APPROACHES


def test_warm_snapshot_name_carries_agent_bs_contract():
    """The warm-up CALLS the planner, so the contract decides what shape the stored
    episodes have. The snapshot is reused whenever it exists, so a name that omits
    the contract silently re-loads a store of per-VNF split exemplars and the change
    reads as having had no effect -- the trap `mb-capacity-and-floor-fix` records for
    the capacity. Empty tag when every flag is off, so the §Y.15 w35/k50 snapshots
    keep their names.
    """
    src = inspect.getsource(G.eval_memory)
    assert "_ctag" in src and "COLOCATION_CONTRACT" in src, (
        "the M^B warm snapshot name does not depend on Agent B's output contract")
    head = src[:src.index("_ctag = ")]
    assert "snap = " not in head, "the name is built before the contract tag exists"


def test_fp_repair_pairing_is_what_it_claims():
    """`Memory-off-fp` carries no repair and `Memory-off-rpg-fp` carries the guard,
    so the pair isolates the h^m guard under a fixed executor."""
    assert G.REPAIR_APPROACHES.get("Memory-off-fp") is None
    assert G.REPAIR_APPROACHES.get("Memory-off-rpg-fp") == "guard"
    assert G.MEMORY_APPROACHES.get("Memory-off-rpg-fp") is None, (
        "an -fp variant of Memory-off must not switch M^B on")


def test_every_approach_has_a_dispatch():
    """No name in APPROACHES may reach the `no eval dispatch` SystemExit.

    `Memory-off-fp` was registered in every table -- stack, agent, mode, repair --
    and still died here, because `run_cell` restated the LLM approaches as a literal
    tuple. This is the third bug of that exact shape (`need_llm`, the dispatch, and
    the checkpoint set), so it gets a test that enumerates rather than a fix.
    """
    import re
    src = inspect.getsource(G.run_cell)
    literal = re.findall(r'approach == "([^"]+)"', src)
    for a in G.APPROACHES:
        dispatched = (a in literal
                      or a in G.NONMEMORY_LLM_APPROACHES
                      or a in G.MEMORY_LLM_APPROACHES)
        assert dispatched, f"{a} is selectable but run_cell has no branch for it"


def test_variant_base_is_transitive():
    assert G._variant_base("Memory-off-rpg-fp") == "Memory-off"
    assert G._variant_base("Full-rp") == "Full"
    assert G._variant_base("RL-alone") == "RL-alone"
