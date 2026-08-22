"""§Y protocol guards: the run must not silently use a superseded configuration.

Every property here is one where the failure is SILENT. A stale scenario, a
superseded checkpoint-selection rule and an evaluation on a spent validation
instance all produce a complete run in which every cell carries a plausible
acceptance number, so nothing surfaces until someone compares tables months
later. That is the class of defect §Y has been paying for repeatedly.
"""

from __future__ import annotations

import inspect
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


def test_a_stack_with_no_registered_probe_is_refused():
    """Picking a selection probe silently would make an arm's meaning depend on
    an undocumented choice. Anything unregistered is refused, not defaulted."""
    assert set(G.SELECTABLE_STACKS) == {"rl_alone", "po_prior", "llm_guided"}

    args = type("A", (), {"passes": 1, "train_instances": None, "lr": 3e-3,
                          "arrivals": 2000, "eval_seg": None,
                          "final_segment": False, "eval_only": True})()
    with pytest.raises(SystemExit, match="does not specify the checkpoint-selection probe"):
        G.select_checkpoint("conventional", 42, "not_a_registered_stack", args)


def test_orion_trains_with_the_planner_in_the_loop():
    """ORION's claim is that the planner guides LEARNING, not only inference, so its
    stack must train with the planner present. Pinned because the planner reached the
    policy through three channels historically and only two carry signal: the plan it
    builds during the curriculum, and advising applied inside the training rollouts."""
    assert G.STACK_FOR_APPROACH["Memory-off"] == "llm_guided"
    assert G.STACK_FOR_APPROACH["Full"] == "llm_guided"
    # The LLM-free control stays on the unadvised stack.
    assert G.STACK_FOR_APPROACH["RL-alone"] == "rl_alone"
    # `Prior-only` is deleted. It meant "the KL prior shaped training, the planner is
    # absent at eval"; with no KL channel it named a policy trained on
    # partial-observability heuristic plans and scored on full-observability greedy
    # ones, with no planner anywhere. Nothing may reintroduce it silently.
    assert "Prior-only" not in G.APPROACHES
    assert "Prior-only" not in G.STACK_FOR_APPROACH

    src = inspect.getsource(G.curriculum_train)
    assert "unknown curriculum config" in src, "no refusal on an unknown config"
    assert 'W.TRAIN_MDO_MODE = "advised_sample"' in src, (
        "llm_guided must train under the advised distribution it evaluates under")
    # Comments stripped: the branch's own comment explains the KL term it omits.
    branch = "\n".join(ln for ln in src.split('config == "llm_guided"')[1]
                       .split("else:")[0].splitlines()
                       if not ln.strip().startswith("#"))
    assert "bs = be = 0.0" in branch, "llm_guided must hold beta at zero"
    assert "BETA_FLOOR" not in branch, "llm_guided must not anneal a KL term"


def test_an_advised_stack_is_selected_and_scored_advised():
    """A checkpoint chosen by scoring the policy in a decode mode it never runs in is
    chosen for the wrong policy. The per-round curve §Y.14 selects on must therefore
    decode the way the stack will be used."""
    assert G.SELECTABLE_STACKS["llm_guided"] == "Memory-off"
    assert G.SELECTABLE_STACKS["llm_guided"] in G.ADVISED_APPROACHES
    import wp7_runner as W
    src = inspect.getsource(W.train_approach)
    # The mode expression moved out of the eval_acceptance call into `_eval_mode`
    # on 2026-08-20 (§9 added a follow_prior branch for actor-only training).
    # The PROPERTY is unchanged and is what this pins: an advised-sample stack
    # must have its per-round curve scored advised, and that curve must be the
    # one handed to eval_acceptance.
    assert '_eval_mode = ("advised" if TRAIN_MDO_MODE == "advised_sample"' in src
    assert "mode=_eval_mode" in src
    # An actor-only stack trains follow_prior and must be scored follow_prior
    # for the same reason: its partition policy was never trained.
    assert 'else "follow_prior" if TRAIN_MDO_MODE == "follow_prior"' in src


def test_the_training_decode_mode_is_reset_per_cell():
    """`TRAIN_MDO_MODE` is module state on wp7_runner. Left set, the next stack in the
    same process would train advised without asking, which is silent and would not
    fail anything."""
    assert inspect.getsource(G._wire).count('W.TRAIN_MDO_MODE = "sample"') == 1


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


def test_wire_pins_the_prior_coupling_loss_and_the_advantage_mode():
    """§Z.7. Both are wp7 module defaults that the grid must override, and the
    failure mode is identical: the run completes, every cell carries a plausible
    number, and nothing says the policy trained under a superseded objective.
    PRIOR_LOSS was the one that got missed, and RL-poprior paid for it in 20/20
    paired cells before anyone looked."""
    import wp7_runner as W

    G._wire("conventional", G.TRAINING_LEVEL, G.TRAIN_INSTANCES[0])
    assert W.PRIOR_LOSS == "distill", (
        "the grid is training with wp7's module default again; sampled_kl is the "
        "term measured as unable to align at any beta")
    assert W.ADV_MODE == "td0"


def test_the_contradictory_prior_term_is_not_what_wire_selects():
    """Pinning the value is not enough on its own: the point is that the legacy
    term is never what a grid cell runs, whatever wp7's default becomes."""
    import wp7_runner as W

    G._wire("conventional", G.TRAINING_LEVEL, G.TRAIN_INSTANCES[0])
    assert W.PRIOR_LOSS != "sampled_kl"


def test_plan_call_bounds_its_completion():
    """The server reserves max_tokens on top of the prompt and refuses the request
    with a 400 if the sum exceeds n_ctx. Leaving the 2048 default in place killed
    both overnight LLM jobs after 11.7 h and 15.3 h, mid eval, once M^B exemplars
    pushed a prompt past ~2000 tokens. A plan is 50-90 tokens."""
    import inspect

    from orion.llm import agent_b as AB

    assert AB.PLAN_MAX_TOKENS <= 512, "a plan does not need a large completion"
    src = inspect.getsource(AB.AgentB)
    assert "max_tokens=PLAN_MAX_TOKENS" in src, (
        "the plan call fell back to the backend's default ceiling")


def test_the_curriculum_branch_is_not_overwritten():
    """`_wire` resets the per-cell knobs, so calling it AFTER the per-config branch
    silently undoes that branch. It was called twice until 2026-08-12, which meant
    CUSTOM_PLAN_BUILDER went back to None and every LLM-free stack trained on
    full-substrate greedy plans while being evaluated on partial-observability ones.

    Nothing failed when that happened. The runs completed and every cell carried a
    plausible acceptance number, which is why this is pinned by position rather than
    by outcome."""
    src = inspect.getsource(G.curriculum_train)
    body = src.split("for i, (level_i, inst_i) in enumerate(segments):")[1]
    assert body.count("_wire(scenario, level_i, inst_i)") == 1, (
        "_wire is called more than once per segment; a call after the config "
        "branch resets CUSTOM_PLAN_BUILDER, RC_FIXED_TRAIN_STREAM and "
        "TRAIN_MDO_MODE back to their defaults")
    # And it must come BEFORE the branch, or the branch is still clobbered.
    assert body.index("_wire(scenario, level_i, inst_i)") < body.index('config == "rl_alone"')


def test_wire_resets_every_knob_the_curriculum_sets():
    """The reset list and the set list must not drift apart. A knob set by a config
    branch but not reset by `_wire` leaks into the next stack trained in the same
    process, which is the same class of silent failure in the other direction."""
    wire = inspect.getsource(G._wire)
    for knob in ("W.CUSTOM_PLAN_BUILDER", "W.RC_FIXED_TRAIN_STREAM", "W.TRAIN_MDO_MODE"):
        assert knob in wire, f"{knob} is set per config but never reset by _wire"


def test_the_ppo_ratio_is_taken_between_two_advised_distributions():
    """PPO's ratio is exp(new_log_prob - old_log_prob). It is only a ratio between
    two settings of ONE policy if both sides carry the same advisory bias. The
    rollout samples from the advised policy, so the recomputation in the update must
    be advised too.

    Recomputing unbiased is silent: the loss still falls and training still finishes,
    but every advised update is computed for a policy that never acted."""
    import wp7_runner as W
    from orion.mdo.policy import AutoregMDOPolicy

    sig = inspect.signature(AutoregMDOPolicy.evaluate_actions).parameters
    assert "prior_logits" in sig and "prior_weight" in sig, (
        "evaluate_actions cannot express the bias the rollout was sampled under")

    src = inspect.getsource(W.train_approach)
    call = src.split("new_lp, new_ent, new_logits = policy.evaluate_actions(")[1][:400]
    assert "prior_logits=pri_i" in call, "the PPO recomputation dropped the bias"
    assert 'TRAIN_MDO_MODE == "advised_sample"' in src, (
        "the bias must be applied exactly when the rollout was advised")
