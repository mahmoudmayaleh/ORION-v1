#!/usr/bin/env python
"""§Y.6 — calibrate RETRIEVAL_FLOOR against the condition key.

Why this exists. `RETRIEVAL_FLOOR` carried the comment "CALIBRATED, NOT ASSUMED
-- see scripts/probe_retrieval.py". That script was not in the tree, so the claim
could not be checked, and the distributions it referred to were same-TOPOLOGY vs
cross-TOPOLOGY. Under §Y the topology is fixed and the state term scores network
CONDITION instead, so even a real prior calibration would have been measuring a
different quantity. The floor has been running as an assumed 0.5.

What a floor has to do. `retrieve` drops candidates scoring below it, so the
planner runs zero-shot when memory has nothing relevant. That only works if the
score actually separates relevant from irrelevant entries:

    combined = LAMBDA * f_task + (1 - LAMBDA) * f_state

A floor set too low never abstains and the planner is always shown something,
relevant or not. Set too high it abstains always and the memory approaches
collapse onto Memory-off, which would read as "memory does not help" when it
means "memory was switched off by a constant".

This probe builds a store of episodes spanning the congestion regimes, then
scores queries drawn from a KNOWN regime against entries from the SAME regime and
from OTHER regimes, and reports the two score distributions plus the separation a
threshold could achieve. It prints a recommended floor and the abstain rate that
floor implies. It does NOT edit the constant: the value goes into the
pre-registration by hand, so the number is a recorded decision rather than a
silently drifting default.

Run:  PYTHONPATH=src python scripts/probe_retrieval_floor.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from orion.llm.condition_signature import (  # noqa: E402
    compute_condition_signature,
    condition_similarity,
)
from orion.llm.episodic_memory import (  # noqa: E402
    RETRIEVAL_FLOOR,
    RETRIEVAL_LAMBDA,
    EpisodicMemory,
)
from orion.sim.slice_generator import generate_slice_request  # noqa: E402
from orion.substrate.hierarchical_topology import (  # noqa: E402
    generate_hierarchical_topology,
)

#: Congestion regimes to span, as the fraction of substrate capacity to consume
#: before snapshotting the condition. Chosen to straddle the bucket boundaries in
#: condition_signature (0.75 / 0.50 / 0.25).
DRAIN_LEVELS = (0.0, 0.35, 0.60, 0.85)


def _drain(substrate, fraction, rng):
    """Consume `fraction` of every node's CPU/RAM, so the condition signature moves.

    Drains uniformly rather than by running a placer: the point is to produce
    known, well-separated congestion states, not a realistic occupancy pattern.
    A realistic pattern would confound the separation measurement with whatever
    the placer happens to prefer.
    """
    for node_id in substrate.graph.nodes:
        attrs = substrate.graph.nodes[node_id]
        jitter = 1.0 + 0.15 * float(rng.standard_normal())
        take = min(0.98, max(0.0, fraction * jitter))
        attrs["cpu_residual"] = attrs["cpu_capacity"] * (1.0 - take)
        attrs["ram_residual"] = attrs["ram_capacity"] * (1.0 - take)
    for _u, _v, d in substrate.graph.edges(data=True):
        d["bw_residual"] = d["bandwidth_capacity"] * (1.0 - fraction)
    return substrate


def _states(instance_seed, rng):
    """One condition signature per congestion regime."""
    out = []
    for i, frac in enumerate(DRAIN_LEVELS):
        sub = generate_hierarchical_topology(instance_seed)
        _drain(sub, frac, rng)
        out.append((f"L{i + 1}", compute_condition_signature(sub, f"L{i + 1}")))
    return out


def _percentiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    def q(p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"min": xs[0], "p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "max": xs[-1]}


def _fmt(d):
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--slices-per-state", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- state term ---------------------------------------------------------
    same, cross = [], []
    for inst in range(args.instances):
        a = _states(inst, rng)
        b = _states(inst + 50, rng)   # a different instance, same regimes
        for (la, ca) in a:
            for (lb, cb) in b:
                (same if la == lb else cross).append(condition_similarity(ca, cb))

    print("=" * 74)
    print("f_state — condition similarity")
    print("=" * 74)
    print(f"  same regime   n={len(same):<5} {_fmt(_percentiles(same))}")
    print(f"  cross regime  n={len(cross):<5} {_fmt(_percentiles(cross))}")
    gap = statistics.median(same) - statistics.median(cross)
    print(f"  median separation: {gap:+.3f}")
    if gap < 0.10:
        print("  !! the state term barely separates regimes; a floor on the combined")
        print("     score cannot recover discrimination the term does not have.")

    # ---- combined score, through the real retriever --------------------------
    # The state term is only half the score. What the floor actually gates is
    # `combined`, so it must be measured through `retrieve`, not reconstructed.
    mem = EpisodicMemory(max_entries=500, write_policy="write_all")
    states = _states(0, rng)
    written = 0
    for label, cond in states:
        sub = generate_hierarchical_topology(0)
        for j in range(args.slices_per_state):
            sr = generate_slice_request(
                request_id=f"{label}_{j:03d}", substrate=sub, rng=rng,
                arrival_time=0.0, lifetime=20.0)
            written += bool(mem.record(
                slice_spec={"request_id": sr.request_id,
                            "slice_type": str(sr.slice_type),
                            "num_vnfs": len(sr.vnfs),
                            "regime": label},
                plan={"domains": [0] * len(sr.vnfs)},
                m_committed=1.0,
                constraints_violated=[],
                reward=1.0,
                plan_shape={"strategy": "colocate", "tiers": ["mec"], "cuts": []},
                condition_signature=cond,
            ))

    matched, mismatched = [], []
    sub = generate_hierarchical_topology(0)
    for label, cond in states:
        for j in range(args.slices_per_state):
            sr = generate_slice_request(
                request_id=f"q_{label}_{j:03d}", substrate=sub, rng=rng,
                arrival_time=0.0, lifetime=20.0)
            hits = mem.retrieve(
                query=f"{sr.slice_type} chain of {len(sr.vnfs)}",
                top_k=5, condition=cond, min_score=0.0)
            for h in hits:
                # The stored regime label rides in the slice_spec, so a hit can be
                # attributed without relying on how content happens to be formatted.
                is_same = f'"regime": "{label}"' in h.entry.content
                (matched if is_same else mismatched).append(h.score)

    print()
    print("=" * 74)
    print(f"combined score (lambda={RETRIEVAL_LAMBDA}) — {written} entries stored")
    print("=" * 74)
    print(f"  same-condition hits  n={len(matched):<5} {_fmt(_percentiles(matched))}")
    print(f"  other-condition hits n={len(mismatched):<5} {_fmt(_percentiles(mismatched))}")

    all_scores = matched + mismatched
    if not all_scores:
        print("\n  NO HITS AT ALL. Retrieval returns nothing regardless of floor;")
        print("  the floor is not the problem and calibrating it would be meaningless.")
        return 1

    print()
    print("  abstain rate as a function of the floor:")
    print(f"    {'floor':>6} {'abstain%':>9} {'same kept':>11} {'other kept':>11}")
    best = None
    for floor in [i / 20 for i in range(21)]:
        kept_same = sum(1 for s in matched if s >= floor)
        kept_other = sum(1 for s in mismatched if s >= floor)
        abstain = 100.0 * (1 - (kept_same + kept_other) / len(all_scores))
        print(f"    {floor:>6.2f} {abstain:>8.1f}% {kept_same:>11} {kept_other:>11}")
        # Youden-style: keep the relevant, drop the rest.
        if matched and mismatched:
            j = kept_same / len(matched) - kept_other / len(mismatched)
            if best is None or j > best[1]:
                best = (floor, j)

    print()
    print("=" * 74)
    print(f"current RETRIEVAL_FLOOR = {RETRIEVAL_FLOOR}")
    if best:
        print(f"recommended floor       = {best[0]:.2f}   (separation J = {best[1]:.3f})")
        if best[1] < 0.2:
            print("  !! weak separation. Report the floor as a design choice, NOT as a")
            print("     calibrated threshold, and say so in the pre-registration.")
    else:
        print("recommended floor       = INDETERMINATE (one class empty)")
    print("Record the chosen value in PREREG_AMENDMENT_2026-07-27_Y §Y.6 by hand.")
    print("This probe deliberately does not edit the constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
