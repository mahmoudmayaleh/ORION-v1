"""§Y protocol guards: the run must not silently use a superseded configuration.

Every property here is one where the failure is SILENT. A stale scenario, a
superseded checkpoint-selection rule and an evaluation on a spent validation
instance all produce a complete run in which every cell carries a plausible
acceptance number, so nothing surfaces until someone compares tables months
later. That is the class of defect §Y has been paying for repeatedly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import grid_runner as G  # noqa: E402


# ── scenario classes ────────────────────────────────────────────────────────

def test_stress_is_not_a_y_scenario():
    """`stress` was the DEFAULT alongside `conventional` until 2026-08-03, so a
    bare `grid_runner.py --part full` produced 80 cells of a scenario the
    registered design does not contain, at full cost and without complaint."""
    assert "stress" not in G.Y_SCENARIOS
    assert G.Y_SCENARIOS == ["conventional", "complex"]


def test_scenario_choices_are_restricted_not_merely_defaulted():
    """A default can be overridden by a stale command line; `choices` cannot."""
    import argparse
    import inspect

    src = inspect.getsource(G.main)
    assert "choices=Y_SCENARIOS" in src, (
        "--scenarios accepts arbitrary values again; a stale invocation can "
        "reintroduce the pre-§Y stress class")
    # And the parser really rejects it.
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=list(G.Y_SCENARIOS),
                    choices=G.Y_SCENARIOS)
    with pytest.raises(SystemExit):
        ap.parse_args(["--scenarios", "stress"])


# ── §Y.14 instance roles ────────────────────────────────────────────────────

def test_validation_and_evaluation_instances_are_disjoint():
    """Selecting a checkpoint on the instance it is reported at is selection on
    test, which is the whole reason §Y.14 splits them."""
    assert G.EVAL_INSTANCE not in G.VALIDATION_INSTANCES
    assert len(G.VALIDATION_INSTANCES) >= 1
    assert set(G.VALIDATION_INSTANCES) & {G.EVAL_INSTANCE} == set()


def test_the_evaluation_instance_is_the_registered_one():
    from orion.substrate.hierarchical_topology import HELDOUT_INSTANCES

    assert G.EVAL_INSTANCE == HELDOUT_INSTANCES[0] == 100
    assert set(G.VALIDATION_INSTANCES) == set(HELDOUT_INSTANCES[1:])


def test_spent_validation_instances_are_refused_for_reporting():
    import inspect

    src = inspect.getsource(G.main)
    assert "VALIDATION_INSTANCES" in src and "spent" in src, (
        "--eval-instances no longer refuses the §Y.14 validation instances")


# ── §Y.14 selection is the DEFAULT, not an opt-in ───────────────────────────

def test_selection_is_the_default_and_the_old_readout_is_the_opt_out():
    """If selection were opt-in, every future run would quietly use the §Y.13
    final-segment readout that missed its pre-registered criterion."""
    import inspect

    src = inspect.getsource(G.get_stacks)
    assert "select_checkpoint" in src
    assert "args.final_segment" in src, "no explicit opt-out; selection cannot be default"

    main_src = inspect.getsource(G.main)
    assert '"--final-segment"' in main_src
    assert "store_true" in main_src, "--final-segment must be a flag, so absence means selection"


def test_llm_stacks_are_refused_rather_than_selected_on_an_unregistered_probe():
    """§Y.14 registers the selection probe only for LLM-free stacks. Picking one
    for an LLM stack silently would make the arm's meaning depend on an
    undocumented choice."""
    assert "llm_prior" not in G.SELECTABLE_STACKS
    assert set(G.SELECTABLE_STACKS) == {"rl_alone", "po_prior"}

    args = type("A", (), {"passes": 1, "train_instances": None, "lr": 3e-3,
                          "arrivals": 2000, "eval_seg": None,
                          "final_segment": False, "eval_only": True})()
    with pytest.raises(SystemExit, match="does not specify the checkpoint-selection probe"):
        G.select_checkpoint("conventional", 42, "llm_prior", args)


def test_every_trained_approach_maps_to_a_stack():
    """A trained approach with no stack mapping would bank cells with no record
    of which weights produced them."""
    assert set(G.STACK_FOR_APPROACH) == G.TRAINED_APPROACHES


# ── curriculum identity ─────────────────────────────────────────────────────

def test_curriculum_segments_cover_the_whole_training_pool():
    """§Y.13: the curriculum drew 4 instances per seed and repeated them, which
    diverged from the registered protocol and dominated the cross-seed spread."""
    from orion.substrate.hierarchical_topology import TRAIN_INSTANCES

    segs = G._curriculum_segments(42)
    assert len(segs) == len(TRAIN_INSTANCES)
    assert {inst for _lvl, inst in segs} == set(TRAIN_INSTANCES)


def test_curriculum_never_trains_on_a_held_out_instance():
    from orion.substrate.hierarchical_topology import HELDOUT_INSTANCES

    for seed in (42, 43, 44, 45, 46):
        trained_on = {inst for _lvl, inst in G._curriculum_segments(seed)}
        assert not trained_on & set(HELDOUT_INSTANCES), (
            f"seed {seed} trains on a held-out instance")


def test_curriculum_order_varies_by_seed_but_the_set_does_not():
    orders = [tuple(i for _l, i in G._curriculum_segments(s)) for s in (42, 43, 44)]
    assert len({frozenset(o) for o in orders}) == 1, "seeds see different instances"
    assert len(set(orders)) > 1, "segment order no longer varies by seed"
