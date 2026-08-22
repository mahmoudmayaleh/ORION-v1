"""§AB — the M^B load-level term is a cliff, and that is why retrieval abstains.

Diagnosed from data/parity_cells: with the store warmed at L2 and frozen, abstain
runs .954/.903/.981 at L1, .027/.000/.019 at L2, and .99+ at L3/L4. It fires at
exactly one level.

Retrieval scores 0.5*f_task + 0.5*f_state against RETRIEVAL_FLOOR = 0.60, and
f_state averages six terms of which one is `1.0 if load_level equal else 0.0`. So
the label alone is worth 0.0833 of the combined score. The L1 near-miss quantiles
are p50 .5805, p90 .5939, max .5999 against a floor of .6000 -- adding that 0.0833
back clears every one of them, which is what says L1's abstain is the label rather
than any real dissimilarity.

These tests pin the arithmetic, because it is the whole argument.
"""

from __future__ import annotations

import pytest

import orion.llm.condition_signature as CS
from orion.llm.episodic_memory import RETRIEVAL_FLOOR, RETRIEVAL_LAMBDA


def _cond(level):
    return dict(cpu_residual_frac=0.55, ram_residual_frac=0.55,
                domain_cpu_residual=[0.5] * 5, domain_ram_residual=[0.5] * 5,
                tier_cpu_residual={"edge": 0.4, "regional_cloud": 0.5,
                                   "central_cloud": 0.95},
                load_level=level)


@pytest.fixture(autouse=True)
def _restore():
    prev = CS.GRADED_LOAD_LEVEL
    yield
    CS.GRADED_LOAD_LEVEL = prev


def test_default_is_the_shipped_cliff():
    """Default must stay off or every banked retrieval number moves."""
    import os
    assert CS.GRADED_LOAD_LEVEL == (os.environ.get("ORION_GRADED_LEVEL", "0") != "0")


def test_level_label_costs_exactly_one_sixth_of_f_state():
    """The number the whole diagnosis rests on: identical conditions differing
    only in the label lose 1/6 of f_state, i.e. 0.0833 of the combined score."""
    CS.GRADED_LOAD_LEVEL = False
    same = CS.condition_similarity(_cond("L2"), _cond("L2"))
    diff = CS.condition_similarity(_cond("L2"), _cond("L3"))
    assert same == pytest.approx(1.0)
    assert diff == pytest.approx(1.0 - 1.0 / 6, abs=1e-6)
    combined_cost = (same - diff) * (1.0 - RETRIEVAL_LAMBDA)
    assert combined_cost == pytest.approx(0.0833, abs=5e-4)


def test_l1_near_misses_clear_the_floor_once_the_cliff_is_removed():
    """Every banked L1 near-miss quantile is below .6000 and above it after the
    label is restored. If this stops holding, the "L1 abstain is an artefact"
    claim is void."""
    CS.GRADED_LOAD_LEVEL = False
    cost = (CS.condition_similarity(_cond("L2"), _cond("L2"))
            - CS.condition_similarity(_cond("L2"), _cond("L1"))) * (1 - RETRIEVAL_LAMBDA)
    for q in (0.5805, 0.5939, 0.5999):        # p50, p90, max, seed 42 L1
        assert q < RETRIEVAL_FLOOR
        assert q + cost > RETRIEVAL_FLOOR


def test_graded_is_a_ramp_not_a_cliff_and_still_orders_the_ladder():
    CS.GRADED_LOAD_LEVEL = True
    warm = _cond("L2")
    s = {lv: CS.condition_similarity(warm, _cond(lv)) for lv in ("L1", "L2", "L3", "L4")}
    assert s["L2"] == pytest.approx(1.0)
    assert s["L1"] == pytest.approx(s["L3"])          # both one step away
    assert s["L3"] > s["L4"]                          # two steps is further
    assert all(v > 0 for v in s.values())             # no cliff to zero
    # An unknown label must NOT become a near neighbour of everything.
    assert CS.condition_similarity(warm, _cond("Lz")) < s["L4"]


def test_graded_lifts_l1_over_the_floor():
    """The point of the change: L1's median near-miss becomes retrievable."""
    CS.GRADED_LOAD_LEVEL = True
    gain = (CS.condition_similarity(_cond("L2"), _cond("L2"))
            - CS.condition_similarity(_cond("L2"), _cond("L1"))) * (1 - RETRIEVAL_LAMBDA)
    assert 0.5805 + (0.0833 - gain) > RETRIEVAL_FLOOR
