"""§AG — the metrics beyond acceptance, and the guarantee that they are populated.

The failure mode these guard is not a wrong number, it is an ABSENT one: every
block is attached inside a bare `except Exception: pass` so instrumentation can
never break an evaluation, which also means a broken block would bank a cell that
looks complete and is silently missing its structure. So the arithmetic is pinned
here and the wiring is pinned by running a real cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import structure_metrics as SM  # noqa: E402


def test_partition_structure_arithmetic():
    out = SM.partition_structure(
        [[0, 0, 0], [0, 1, 1], [2, 2], [0, 1, 2]], domain_ids=[0, 1, 2, 3])
    assert out["n_accepted"] == 4
    assert out["split_rate"] == 0.5             # two of four use >1 domain
    assert out["domains_per_chain_mean"] == 1.75  # (1 + 2 + 1 + 3) / 4
    assert out["domains_per_chain_max"] == 3
    assert out["domains_per_chain_hist"] == {"1": 2, "2": 1, "3": 1}
    assert out["noncontiguous_accepted"] == 0
    # 5 VNFs in d0, 3 in d1, 3 in d2, none in d3: the empty domain must be counted
    assert out["vnfs_per_domain"] == {"0": 5, "1": 3, "2": 3, "3": 0}
    assert 0.0 < out["domain_load_jain"] < 1.0


def test_jain_extremes():
    assert SM._jain([5, 5, 5, 5]) == 1.0            # perfectly even
    assert SM._jain([10, 0, 0, 0]) == 0.25          # one domain does everything
    assert SM._jain([0, 0]) is None                 # nothing admitted


def test_noncontiguous_is_counted_not_hidden():
    out = SM.partition_structure([[0, 1, 0]], domain_ids=[0, 1])
    assert out["noncontiguous_accepted"] == 1, (
        "an accepted partition that re-enters a domain must be visible; C10 should "
        "make this zero, and a non-zero value means the constraint leaked")


def test_qos_margin():
    out = SM.qos_margin([(5.0, 10.0), (9.0, 10.0), (20.0, 10.0)])
    assert out["n"] == 3
    assert out["budget_ratio_mean"] == round((0.5 + 0.9 + 2.0) / 3, 4)
    assert out["over_budget_frac"] == round(1 / 3, 4)


def test_qos_margin_drops_broken_budgets():
    out = SM.qos_margin([(5.0, 0.0), (float("inf"), 10.0), (4.0, 8.0)])
    assert out["n"] == 1, "a zero or infinite record must be dropped, not clamped"


def test_acceptance_windows():
    out = SM.acceptance_windows([True] * 100 + [False] * 100, window=100)
    assert out["acceptance"] == [1.0, 0.0]
    assert out["first_window"] == 1.0 and out["last_window"] == 0.0


def test_empty_inputs_are_reported_not_guessed():
    assert SM.partition_structure([])["n_accepted"] == 0
    assert SM.qos_margin([])["n"] == 0
    assert SM.acceptance_windows([])["acceptance"] == []


# ── wiring: a real cell must come back with all three blocks ───────────────────
def test_blocks_are_banked_by_the_pipeline_path():
    import grid_runner as G

    out = G.eval_plain_partial("conventional", 42, "L3", 100, 120)
    cost = out["cost"]
    for block in ("structure", "qos", "timeseries"):
        assert block in cost, f"{block} missing from a pipeline cell"
    assert cost["structure"]["n_accepted"] > 0
    assert cost["qos"]["n"] > 0
    assert cost["structure"]["noncontiguous_accepted"] == 0, (
        "C10 is on, so no accepted partition may re-enter a domain")


def test_blocks_are_banked_by_the_plain_path():
    import grid_runner as G

    out = G.eval_plain("conventional", 42, "L3", 100, 120)
    cost = out["cost"]
    for block in ("structure", "qos", "timeseries"):
        assert block in cost, f"{block} missing from the Plain cell"
    assert cost["structure"]["n_accepted"] > 0
