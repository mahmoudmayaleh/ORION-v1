"""Property guard for M^B composition telemetry (bug 1, 2026-07-16).

`_mb_composition` reported mb_pos=0 on every run ever recorded, because it read
`e.admitted` / `e.success` -- attributes MemoryEntry does not define -- and fell
through to the False default. The failure was silent: "mb=50(+0/-50)" looked like
a finding ("M^B learned only from failures") rather than a broken read.

These tests fail if the telemetry ever again cannot express BOTH states. The
canary is the point: a field that can only ever report one value is not measuring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from r_local_runner import _fresh_mb, _mb_composition


def _write(mb, admitted: bool, rid: str):
    """Write one episode the way run_q_cell does (reward +1 admit / -1 reject)."""
    violations = [] if admitted else ["cross_domain_bw"]
    mb.record(
        slice_spec={"slice_type": "eMBB", "num_vnfs": 3, "request_id": rid},
        plan={"placements": []},
        m_committed=0.0,
        constraints_violated=violations,
        reward=1.0 if admitted else -1.0,
        topology_signature={"sig": "test"},
        plan_shape=None,
        violation_tag=violations[0] if violations else None,
    )


def test_composition_counts_positives_and_negatives():
    """The regression: 2 admits + 1 reject must read as (+2/-1), not (+0/-3)."""
    mb = _fresh_mb()
    _write(mb, True, "r1")
    _write(mb, True, "r2")
    _write(mb, False, "r3")

    c = _mb_composition(mb)
    assert c["mb_entries"] == 3, c
    assert c["mb_pos"] == 2, f"positives invisible again (bug 1 regression): {c}"
    assert c["mb_neg"] == 1, c


def test_all_positive_memory_never_reads_as_all_negative():
    """The exact shape of the original bug: an all-admit stream reported +0."""
    mb = _fresh_mb()
    for i in range(5):
        _write(mb, True, f"r{i}")

    c = _mb_composition(mb)
    assert c["mb_pos"] == 5, f"all-success M^B read as {c} -- telemetry is dead"
    assert c["mb_neg"] == 0, c


def test_empty_memory_is_zero_not_crash():
    c = _mb_composition(_fresh_mb())
    assert c == {"mb_entries": 0, "mb_pos": 0, "mb_neg": 0}, c


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
