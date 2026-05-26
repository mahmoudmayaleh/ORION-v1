"""Tests for EpisodicMemory (M^B) selective recording and retrieval."""

from __future__ import annotations

import pytest

from orion.llm.episodic_memory import EpisodicMemory
from orion.retrieval import RetrievalConfig, RetrievalMode, ScoredEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> RetrievalConfig:
    """Build a KEYWORD_ONLY config that needs no embeddings."""
    return RetrievalConfig(mode=RetrievalMode.KEYWORD_ONLY)


def _embb_slice(idx: int = 1) -> dict:
    return {"slice_type": "eMBB", "user_count": 50 + idx, "qos_class": 1}


def _urllc_slice(idx: int = 1) -> dict:
    return {"slice_type": "URLLC", "user_count": 10 + idx, "qos_class": 2}


def _plan(rb: int = 10) -> dict:
    return {"rb_alloc": rb, "power": 0.5}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem() -> EpisodicMemory:
    """EpisodicMemory with keyword-only pipeline and small capacity."""
    return EpisodicMemory(config=_make_config(), embed_fn=None, max_entries=500)


@pytest.fixture()
def small_mem() -> EpisodicMemory:
    """EpisodicMemory with very small capacity for eviction tests."""
    return EpisodicMemory(config=_make_config(), embed_fn=None, max_entries=20)


# ---------------------------------------------------------------------------
# 1. Selective write -- high reward
# ---------------------------------------------------------------------------

def test_record_high_reward_stored(mem: EpisodicMemory) -> None:
    result = mem.record(
        slice_spec=_embb_slice(),
        plan=_plan(),
        m_committed=5.0,
        constraints_violated=[],
        reward=0.9,
    )
    assert result is True
    assert len(mem._entries) == 1


# ---------------------------------------------------------------------------
# 2. Selective write -- constraint violation
# ---------------------------------------------------------------------------

def test_record_constraint_violation_stored(mem: EpisodicMemory) -> None:
    result = mem.record(
        slice_spec=_embb_slice(),
        plan=_plan(),
        m_committed=5.0,
        constraints_violated=["C4"],
        reward=0.2,
    )
    assert result is True
    assert len(mem._entries) == 1


# ---------------------------------------------------------------------------
# 3. Selective write -- low (negative) reward
# ---------------------------------------------------------------------------

def test_record_low_reward_stored(mem: EpisodicMemory) -> None:
    result = mem.record(
        slice_spec=_embb_slice(),
        plan=_plan(),
        m_committed=5.0,
        constraints_violated=[],
        reward=-0.6,
    )
    assert result is True
    assert len(mem._entries) == 1


# ---------------------------------------------------------------------------
# 4. Mediocre reward skipped
# ---------------------------------------------------------------------------

def test_record_mediocre_reward_skipped(mem: EpisodicMemory) -> None:
    result = mem.record(
        slice_spec=_embb_slice(),
        plan=_plan(),
        m_committed=5.0,
        constraints_violated=[],
        reward=0.3,
    )
    assert result is False
    assert len(mem._entries) == 0


# ---------------------------------------------------------------------------
# 5. Label "success" for high reward, no violations
# ---------------------------------------------------------------------------

def test_record_labels_success_correctly(mem: EpisodicMemory) -> None:
    mem.record(
        slice_spec=_embb_slice(),
        plan=_plan(),
        m_committed=5.0,
        constraints_violated=[],
        reward=0.9,
    )
    entry = mem._entries[0]
    assert entry.tags["label"] == ["success"]
    assert "success" in entry.topic


# ---------------------------------------------------------------------------
# 6. Label "failure" when violations are present
# ---------------------------------------------------------------------------

def test_record_labels_failure_correctly(mem: EpisodicMemory) -> None:
    mem.record(
        slice_spec=_embb_slice(),
        plan=_plan(),
        m_committed=5.0,
        constraints_violated=["C4"],
        reward=0.2,
    )
    entry = mem._entries[0]
    assert entry.tags["label"] == ["failure"]
    assert "failure" in entry.topic


# ---------------------------------------------------------------------------
# 7. Retrieve returns ScoredEntry instances
# ---------------------------------------------------------------------------

def test_retrieve_returns_scored_entries(mem: EpisodicMemory) -> None:
    mem.record(_embb_slice(1), _plan(10), 5.0, [], 0.9)
    mem.record(_embb_slice(2), _plan(20), 8.0, ["C4"], 0.1)
    mem.record(_urllc_slice(1), _plan(5), 3.0, [], 0.85)

    results = mem.retrieve("eMBB placement", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert isinstance(item, ScoredEntry)
        assert hasattr(item, "score")
        assert hasattr(item, "entry")


# ---------------------------------------------------------------------------
# 8. Retrieve with slice_type filter
# ---------------------------------------------------------------------------

def test_retrieve_with_slice_type_filter(mem: EpisodicMemory) -> None:
    mem.record(_embb_slice(1), _plan(10), 5.0, [], 0.9)
    mem.record(_urllc_slice(1), _plan(5), 3.0, [], 0.85)

    results = mem.retrieve(
        "placement plan",
        filters={"slice_type": "URLLC"},
        top_k=5,
    )
    for item in results:
        assert "URLLC" in item.entry.tags.get("slice_type", [])


# ---------------------------------------------------------------------------
# 9. format_for_prompt labels success and failure
# ---------------------------------------------------------------------------

def test_format_for_prompt_success_and_failure(mem: EpisodicMemory) -> None:
    mem.record(_embb_slice(1), _plan(10), 5.0, [], 0.9)
    mem.record(_embb_slice(2), _plan(20), 8.0, ["C4"], 0.1)

    results = mem.retrieve("eMBB placement", top_k=5)
    text = mem.format_for_prompt(results)

    assert "[Positive Example]" in text or "[Counter-Example]" in text
    assert "Past Plans" in text


# ---------------------------------------------------------------------------
# 10. format_for_prompt with empty list
# ---------------------------------------------------------------------------

def test_format_for_prompt_empty(mem: EpisodicMemory) -> None:
    assert mem.format_for_prompt([]) == ""


# ---------------------------------------------------------------------------
# 11. Eviction removes least important entries
# ---------------------------------------------------------------------------

def test_eviction_removes_least_important(small_mem: EpisodicMemory) -> None:
    max_cap = small_mem._max_entries
    # Fill past capacity
    for i in range(max_cap + 5):
        small_mem.record(
            _embb_slice(i),
            _plan(i),
            float(i),
            [],
            0.9,
        )
    # Eviction should have trimmed entries back below max + 1
    assert len(small_mem._entries) <= max_cap


# ---------------------------------------------------------------------------
# 12. Save and load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path, mem: EpisodicMemory) -> None:
    mem.record(_embb_slice(1), _plan(10), 5.0, [], 0.9)
    mem.record(_urllc_slice(1), _plan(5), 3.0, ["C4"], 0.1)

    save_path = tmp_path / "episodic.json"
    mem.save(save_path)

    fresh = EpisodicMemory(config=_make_config(), embed_fn=None)
    fresh.load(save_path)

    assert len(fresh._entries) == 2
    topics = [e.topic for e in fresh._entries]
    assert any("eMBB" in t for t in topics)
    assert any("URLLC" in t for t in topics)


# ---------------------------------------------------------------------------
# 13. Load from nonexistent path is a no-op
# ---------------------------------------------------------------------------

def test_load_nonexistent_path_is_noop(tmp_path, mem: EpisodicMemory) -> None:
    missing = tmp_path / "does_not_exist.json"
    mem.load(missing)
    assert len(mem._entries) == 0


# ---------------------------------------------------------------------------
# 14. Record populates tags correctly
# ---------------------------------------------------------------------------

def test_record_populates_tags(mem: EpisodicMemory) -> None:
    mem.record(
        slice_spec=_urllc_slice(),
        plan=_plan(),
        m_committed=3.0,
        constraints_violated=["C2"],
        reward=0.1,
    )
    entry = mem._entries[0]
    assert "label" in entry.tags
    assert "slice_type" in entry.tags
    assert entry.tags["slice_type"] == ["URLLC"]
    assert entry.tags["label"] == ["failure"]
