"""§AI -- the h^m guard on the FINAL partition.

The property that matters most is NEGATIVE: the guard must not touch a partition
that already passes, because that is what keeps it off the heuristic baselines.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from orion.mdo import admissibility as A  # noqa: E402


class _Sum:
    def __init__(self, did, cpu_res, ram_res, hcpu, hram):
        self.domain_id = did
        self.cpu_residual = cpu_res
        self.ram_residual = ram_res
        self.tier_max_node_cpu = hcpu
        self.tier_max_node_ram = hram


class _VNF:
    def __init__(self, vid, cpu, ram, nodes):
        self.vnf_id, self.cpu_demand, self.ram_demand = vid, cpu, ram
        self.permitted_nodes = nodes


class _SR:
    def __init__(self, vnfs):
        self.vnfs = vnfs


class _Sub:
    """Two domains, one node each, both EDGE."""

    def __init__(self, caps):
        import networkx as nx
        self.graph = nx.Graph()
        self._dom = {}
        for n, (dom, cpu, ram) in caps.items():
            self.graph.add_node(n, tier="edge", domain_id=dom,
                                cpu_residual=cpu, ram_residual=ram)
            self._dom.setdefault(dom, []).append(n)

    def nodes_in_domain(self, d):
        return list(self._dom.get(d, []))


def _fixture(d1_hcpu):
    sub = _Sub({"n0": (0, 100.0, 100.0), "n1": (1, d1_hcpu, d1_hcpu)})
    sums = [_Sum(0, 100.0, 100.0, {"edge": 100.0}, {"edge": 100.0}),
            _Sum(1, d1_hcpu, d1_hcpu, {"edge": d1_hcpu}, {"edge": d1_hcpu})]
    sr = _SR([_VNF("v0", 10.0, 10.0, ["n0", "n1"]),
              _VNF("v1", 10.0, 10.0, ["n0", "n1"])])
    return sub, sums, sr


def test_admissible_partition_is_returned_UNCHANGED():
    """The guarantee that keeps the guard off the baselines."""
    sub, sums, sr = _fixture(100.0)
    A.reset_guard_stats()
    part = [0, 0]
    out = A.repair_partition(part, sr, sub, sums)
    assert out is part, "an admissible partition must be returned by identity"
    assert A.GUARD_STATS["vnfs_moved"] == 0
    assert A.GUARD_STATS["partitions_changed"] == 0


def test_inadmissible_vnf_is_reseated():
    # D1's only node is far too small for a 10-unit VNF: h^m refuses it.
    sub, sums, sr = _fixture(1.0)
    A.reset_guard_stats()
    out = A.repair_partition([1, 1], sr, sub, sums)
    assert out == [0, 0], "both VNFs should move to the only domain that fits"
    assert A.GUARD_STATS["vnfs_moved"] == 2


def test_unrepairable_vnf_is_left_exactly_as_authored():
    """The guard converts rejections; it never invents a placement."""
    sub = _Sub({"n0": (0, 1.0, 1.0), "n1": (1, 1.0, 1.0)})
    sums = [_Sum(0, 1.0, 1.0, {"edge": 1.0}, {"edge": 1.0}),
            _Sum(1, 1.0, 1.0, {"edge": 1.0}, {"edge": 1.0})]
    sr = _SR([_VNF("v0", 50.0, 50.0, ["n0", "n1"])])
    A.reset_guard_stats()
    out = A.repair_partition([1], sr, sub, sums)
    assert out == [1], "nothing fits, so the authored choice must survive"
    assert A.GUARD_STATS["unrepairable"] == 1
    assert A.GUARD_STATS["vnfs_moved"] == 0


def test_repair_never_introduces_a_c10_violation():
    """A repair that reopened a left domain would trade one bin for another."""
    sub = _Sub({"n0": (0, 100.0, 100.0), "n1": (1, 100.0, 100.0)})
    sums = [_Sum(0, 100.0, 100.0, {"edge": 100.0}, {"edge": 100.0}),
            _Sum(1, 100.0, 100.0, {"edge": 100.0}, {"edge": 100.0})]
    # v1 is pinned to D1's node; v0 and v2 may go anywhere.
    sr = _SR([_VNF("v0", 1.0, 1.0, ["n0", "n1"]),
              _VNF("v1", 1.0, 1.0, ["n1"]),
              _VNF("v2", 1.0, 1.0, ["n0", "n1"])])
    A.reset_guard_stats()
    out = A.repair_partition([0, 1, 1], sr, sub, sums)
    runs = sum(1 for i, d in enumerate(out) if i == 0 or out[i - 1] != d)
    assert runs == len(set(out)), f"partition {out} is not chain-contiguous"


def test_length_mismatch_is_a_noop():
    sub, sums, sr = _fixture(100.0)
    part = [0]
    assert A.repair_partition(part, sr, sub, sums) is part


# ── the exact minimal repair (C10 is a joint constraint) ──────────────────────
def _trap():
    """The measured eMBB trap: VNF0 anywhere, VNF1 not in D1, VNF2 ONLY in D1."""
    sub = _Sub({"n0": (0, 100.0, 100.0), "n1": (1, 100.0, 100.0)})
    sums = [_Sum(0, 100.0, 100.0, {"edge": 100.0}, {"edge": 100.0}),
            _Sum(1, 100.0, 100.0, {"edge": 100.0}, {"edge": 100.0})]
    sr = _SR([_VNF("v0", 1.0, 1.0, ["n0", "n1"]),
              _VNF("v1", 1.0, 1.0, ["n0"]),
              _VNF("v2", 1.0, 1.0, ["n1"])])
    return sub, sums, sr


def test_exact_repair_fixes_the_ABA_shape_a_forward_pass_cannot():
    sub, sums, sr = _trap()
    A.reset_guard_stats()
    out = A.minimal_committable_partition([1, 0, 1], sr, sub, sums)
    assert out == [0, 0, 1], f"expected the one-edit fix, got {out}"
    runs = sum(1 for i, d in enumerate(out) if i == 0 or out[i - 1] != d)
    assert runs == len(set(out))


def test_forward_pass_CANNOT_fix_it_which_is_why_the_search_exists():
    """Pins the motivation: the cheap repair genuinely fails on this shape."""
    sub, sums, sr = _trap()
    A.reset_guard_stats()
    out = A.repair_partition([1, 0, 1], sr, sub, sums)
    runs = sum(1 for i, d in enumerate(out) if i == 0 or out[i - 1] != d)
    assert runs != len(set(out)), (
        "the forward pass unexpectedly fixed the trap; if it can, the search is "
        "no longer justified and this test should be revisited rather than deleted")


def test_exact_repair_is_identity_on_an_already_valid_partition():
    sub, sums, sr = _trap()
    A.reset_guard_stats()
    part = [0, 0, 1]
    assert A.minimal_committable_partition(part, sr, sub, sums) is part
    assert A.GUARD_STATS["partitions_changed"] == 0


def test_exact_repair_prefers_the_FEWEST_edits():
    sub, sums, sr = _trap()
    out = A.minimal_committable_partition([0, 0, 0], sr, sub, sums)
    # v2 must move to D1; v0 and v1 were already fine and must be left alone.
    assert out == [0, 0, 1], f"repair changed more than it had to: {out}"


def test_exact_repair_returns_input_when_nothing_is_committable():
    sub = _Sub({"n0": (0, 1.0, 1.0), "n1": (1, 1.0, 1.0)})
    sums = [_Sum(0, 1.0, 1.0, {"edge": 1.0}, {"edge": 1.0}),
            _Sum(1, 1.0, 1.0, {"edge": 1.0}, {"edge": 1.0})]
    sr = _SR([_VNF("v0", 99.0, 99.0, ["n0", "n1"])])
    A.reset_guard_stats()
    part = [0]
    assert A.minimal_committable_partition(part, sr, sub, sums) is part
    assert A.GUARD_STATS["unrepairable"] == 1
