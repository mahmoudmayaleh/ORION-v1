"""use_mb=False must mean NO M^B object reaches the plan builder.

The silent-no-op class: M^B is wired through a `lambda: mb` closure, so it is not
visibly passed at the call site. If a later edit rebinds `mb` after the closure is
built, Track D would silently run the worst-known configuration (§M M.1: -10..-13
admits/100) while its result JSON still claimed mb=None. This pins it.

Renamed from `test_train_arm_use_mb.py` under the §Y.8 vocabulary migration
(`arm` -> `approach`). The old file kept asserting against `W.train_arm` after the
rename, so this guard was silently failing rather than guarding anything, which is
the exact failure mode it was written to prevent.
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_use_mb_param_exists_and_defaults_true():
    import wp7_runner as W
    sig = inspect.signature(W.train_approach)
    assert "use_mb" in sig.parameters, "train_approach lost its use_mb parameter"
    assert sig.parameters["use_mb"].default is True, (
        "use_mb must default True so R45's as-run behavior stays reproducible")


def test_mb_construction_is_gated_on_use_mb():
    import wp7_runner as W
    src = inspect.getsource(W.train_approach)
    gate = 'if approach == "LLM+RL-full" and use_mb:'
    assert gate in src, (
        "the M^B construction is no longer gated on use_mb -- Track D would run "
        "M^B-live while reporting mb=None")
    # mb must be bound to None before the gate and never rebound afterwards.
    before, after = src.split(gate, 1)
    assert "mb = None" in before, "mb is not defaulted to None before the gate"
    tail = after.split("EpisodicMemory(", 1)[1] if "EpisodicMemory(" in after else after
    assert "mb = " not in tail.split("def ")[0].replace("mb = None", ""), (
        "mb is rebound after the gate -- the closure would capture a live M^B")


def test_no_arm_named_entry_point_survives():
    """§Y.8: `arm` is banned in code. A leftover alias would let a stale test pass
    against a name the pipeline no longer calls."""
    import wp7_runner as W
    assert not hasattr(W, "train_arm"), (
        "wp7_runner.train_arm is back; §Y.8 renamed it to train_approach")
