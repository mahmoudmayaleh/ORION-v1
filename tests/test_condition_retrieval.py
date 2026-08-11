"""Guards for the §Y.6 M^B retrieval state key.

These pin a property that can silently do nothing: under a fixed topology the
old state term is constant, so retrieval degrades to text-only ranking AND the
abstain floor becomes unreachable — with no error, no warning, and a plausible
-looking abstain_rate of 0.0 in the telemetry. The first test asserts the
failure mode exists (so nobody "fixes" it back), the rest assert the condition
key actually discriminates.
"""

from __future__ import annotations

import pytest

from orion.llm.condition_signature import (
    condition_query_terms,
    condition_similarity,
    occupancy_bucket,
)
from orion.llm.episodic_memory import (
    RETRIEVAL_FLOOR,
    RETRIEVAL_LAMBDA,
    EpisodicMemory,
)

FREE = {
    "load_level": "L1",
    "cpu_residual_frac": 0.95,
    "ram_residual_frac": 0.93,
    "domain_cpu_residual": [0.95, 0.96, 0.94, 0.95, 0.95],
    "domain_ram_residual": [0.93, 0.94, 0.92, 0.93, 0.93],
    "tier_cpu_residual": {"edge": 0.95, "edge": 0.94,
                          "regional_cloud": 0.96, "central_cloud": 0.97},
    "inter_bw_residual_mean": 0.98,
    "inter_bw_residual_min": 0.96,
    "active_slices": 2,
    "bucket": "free",
}

SATURATED = {
    "load_level": "L4",
    "cpu_residual_frac": 0.08,
    "ram_residual_frac": 0.11,
    "domain_cpu_residual": [0.05, 0.09, 0.07, 0.10, 0.08],
    "domain_ram_residual": [0.10, 0.12, 0.09, 0.13, 0.11],
    # Access saturates first, the core keeps headroom — the hierarchy's
    # characteristic scarcity pattern, and what a whole-network average hides.
    "tier_cpu_residual": {"edge": 0.02, "edge": 0.05,
                          "regional_cloud": 0.15, "central_cloud": 0.45},
    "inter_bw_residual_mean": 0.21,
    "inter_bw_residual_min": 0.04,
    "active_slices": 41,
    "bucket": "saturated",
}

FIXED_TOPOLOGY = {
    "num_domains": 5,
    "nodes_per_domain": [4, 7, 7, 7, 7],
    "tier_coverage": [0.5, 0.75, 0.75, 0.75, 0.75],
    "inter_bw_mean": 600.0,
    "num_inter_links": 8,
}


def _slice_spec(slice_type: str = "urllc") -> dict:
    return {"slice_type": slice_type, "num_vnfs": 3, "request_id": "req_test"}


def _store(condition: dict | None, topology: dict | None,
           slice_type: str = "urllc") -> EpisodicMemory:
    mb = EpisodicMemory(max_entries=50, write_policy="write_all")
    mb.record(
        slice_spec=_slice_spec(slice_type),
        plan={"vnf_assignments": [{"vnf_id": "v0", "domain": 1}]},
        m_committed=0.0,
        constraints_violated=[],
        reward=1.0,
        topology_signature=topology,
        plan_shape={"strategy": "co-locate"},
        condition_signature=condition,
    )
    return mb


# ── The failure mode this change exists to remove ────────────────────────────


def test_topology_key_is_a_no_op_on_a_fixed_substrate():
    """With one topology every entry scores f_state = 1.0, so the state term adds
    the same constant to every candidate and cannot rank them. This is the §Y.6
    blocker, asserted so it cannot be reintroduced by pointing the state term
    back at topology."""
    from orion.llm.episodic_memory import _topology_similarity

    assert _topology_similarity(FIXED_TOPOLOGY, FIXED_TOPOLOGY) == pytest.approx(1.0)
    # Two draws of the same fixed topology are indistinguishable to this term,
    # which is the whole defect. Asserted on the term itself rather than on its
    # arithmetic against a particular floor, since the floor is a calibrated
    # value that moves (0.5 -> 0.60 in §Y.6).
    perturbed = {**FIXED_TOPOLOGY, "inter_bw_mean": 590.0}
    assert _topology_similarity(FIXED_TOPOLOGY, perturbed) == pytest.approx(1.0, abs=0.05)
    # At the pre-§Y.6 floor of 0.5 the constant alone also cleared the threshold,
    # which is what made abstention structurally unreachable rather than merely
    # uninformative.
    assert (1.0 - RETRIEVAL_LAMBDA) * 1.0 >= 0.5


# ── The replacement key ──────────────────────────────────────────────────────


def test_condition_similarity_is_reflexive_and_separating():
    assert condition_similarity(FREE, FREE) == pytest.approx(1.0)
    assert condition_similarity(SATURATED, SATURATED) == pytest.approx(1.0)
    # Opposite operating regimes must not look alike, or the term cannot rank.
    assert condition_similarity(FREE, SATURATED) < 0.4


def test_condition_similarity_is_ordered_by_congestion_distance():
    mid = dict(SATURATED, load_level="L2",
               cpu_residual_frac=0.5, ram_residual_frac=0.5,
               domain_cpu_residual=[0.5] * 5, domain_ram_residual=[0.5] * 5,
               tier_cpu_residual={k: 0.5 for k in SATURATED["tier_cpu_residual"]},
               inter_bw_residual_mean=0.5, inter_bw_residual_min=0.5)
    assert (condition_similarity(SATURATED, mid)
            > condition_similarity(SATURATED, FREE))


def test_missing_condition_scores_zero_not_one():
    """A pre-§Y entry (no Condition line) must not be scored as a perfect state
    match — that would rank legacy entries above every correctly-keyed one."""
    assert condition_similarity(SATURATED, {}) == 0.0


def test_retrieval_prefers_the_matching_congestion_regime():
    mb = EpisodicMemory(max_entries=50, write_policy="write_all")
    for cond, rid in ((FREE, "free"), (SATURATED, "sat")):
        mb.record(
            slice_spec={"slice_type": "urllc", "num_vnfs": 3, "request_id": rid},
            plan={"vnf_assignments": [{"vnf_id": "v0", "domain": 1}]},
            m_committed=0.0,
            constraints_violated=[],
            reward=1.0,
            topology_signature=FIXED_TOPOLOGY,
            plan_shape={"strategy": "co-locate"},
            condition_signature=cond,
        )

    got = mb.retrieve("urllc placement", top_k=2, condition=SATURATED,
                      min_score=None)
    assert got, "retrieval returned nothing"
    assert "sat" in got[0].entry.content, (
        "the free-network episode outranked the saturated one under a "
        "saturated query — the state term is not discriminating")


def test_abstention_is_reachable_under_the_condition_key():
    """The whole point of the floor: a store holding only free-network episodes
    must be able to decline to answer a saturated-network query."""
    mb = _store(FREE, FIXED_TOPOLOGY)
    got = mb.retrieve("urllc placement", top_k=3, condition=SATURATED)
    assert got == []
    stats = mb.retrieval_stats()
    assert stats["abstain_rate"] == 1.0

    # ... and the same store answers a matching query.
    mb.reset_retrieval_log()
    assert mb.retrieve("urllc placement", top_k=3, condition=FREE)


def test_condition_tags_are_filterable():
    mb = _store(SATURATED, FIXED_TOPOLOGY)
    entry = mb._entries[0]
    assert entry.tags["congestion"] == ["saturated"]
    assert entry.tags["load_level"] == ["L4"]


def test_bucket_boundaries():
    assert occupancy_bucket(1.0) == "free"
    assert occupancy_bucket(0.75) == "free"
    assert occupancy_bucket(0.74) == "moderate"
    assert occupancy_bucket(0.50) == "moderate"
    assert occupancy_bucket(0.49) == "tight"
    assert occupancy_bucket(0.25) == "tight"
    assert occupancy_bucket(0.0) == "saturated"


def test_query_terms_name_the_scarce_tier():
    terms = condition_query_terms(SATURATED)
    assert "congestion saturated" in terms
    assert "load L4" in terms
    assert "edge exhausted" in terms
    assert "central_cloud exhausted" not in terms


# ── What the floor turned away ───────────────────────────────────────────────


def test_abstain_records_the_score_it_turned_away():
    """`best_available` is the only evidence for where the floor should sit, and
    it is read on exactly the calls that abstain. It used to be computed after
    the floor filter, so it was 0.0 on every one of them: the §Y.15 grid abstained
    on 78-99% of consultations and recorded nothing about how close any of them
    came. Nothing asserted the field, which is why it stayed dead."""
    mb = _store(FREE, None)
    hits = mb.retrieve("urllc slice", condition=SATURATED)

    assert hits == []                       # opposite regime, floor holds
    log = mb._retrieval_log[-1]
    assert log["abstained"] is True
    assert log["max"] is None               # nothing was returned
    assert log["floor"] == RETRIEVAL_FLOOR
    # The entry scored SOMETHING. A zero here means the field is being read after
    # the filter again and the floor cannot be re-calibrated from a run.
    assert log["best_available"] is not None
    assert 0.0 < log["best_available"] < RETRIEVAL_FLOOR


def test_retrieval_stats_report_the_near_miss_distribution():
    """An abstain rate says the floor fired; the near-miss quantiles say by how
    much. Re-reading the operating point off a real stream needs the second."""
    mb = _store(FREE, None)
    for _ in range(4):
        mb.retrieve("urllc slice", condition=SATURATED)

    stats = mb.retrieval_stats()
    assert stats["calls"] == 4
    assert stats["abstain_rate"] == 1.0
    assert stats["near_miss_n"] == 4
    assert stats["floor"] == RETRIEVAL_FLOOR
    for q in ("near_miss_p50", "near_miss_p90", "near_miss_max"):
        assert 0.0 < stats[q] < RETRIEVAL_FLOOR
    assert stats["near_miss_p50"] <= stats["near_miss_max"]


def test_near_miss_fields_are_absent_when_nothing_abstains():
    mb = _store(SATURATED, None)
    mb.retrieve("urllc slice", condition=SATURATED)   # same regime, fires

    stats = mb.retrieval_stats()
    assert stats["abstain_rate"] == 0.0
    assert stats["near_miss_n"] == 0
    assert stats["near_miss_p50"] is None
