#!/usr/bin/env python3
"""Shift-schedule diagnostic — does a candidate post-shift regime MOVE THE OPTIMUM?

A shift is only valid for the adaptation axis if it changes what the correct policy
IS (partitioning pattern), not just the load. This measures, on C+_T-_B-, the
per-slice-type coloc_feasible / partition_needed statistics (Probe-B classifier),
so a MIX shift can be built from types that actually force partitioning. It also
scores three candidate mixes (baseline / partition-heavy / coloc-heavy) end-to-end.

Output feeds the shift-protocol pre-registration draft: pick segment mixes whose
partition_needed differs materially from baseline, and the change points between them
become the pre-registered shift. No training, no LLM. Minutes.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import five_arm_runner as R
from probe_b_partition_regime import classify_slice
from orion.sim.slice_generator import generate_slice_request
from orion.types import SliceType

FAMILY = "C+_T-_B-"
N_PER = 80


def classify_stream(sub, slice_type, n, seed):
    rng = np.random.default_rng(seed)
    agg = defaultdict(int)
    for i in range(n):
        sr = generate_slice_request(request_id=f"d_{i:04d}", substrate=sub, rng=rng,
                                    slice_type=slice_type)
        c = classify_slice(sr, sub)
        for k, v in c.items():
            agg[k] += int(v)
    return agg


def classify_mix(sub, weights, n, seed):
    rng = np.random.default_rng(seed)
    types = list(weights.keys()); w = np.array([weights[t] for t in types], float); w /= w.sum()
    agg = defaultdict(int)
    for i in range(n):
        st = types[rng.choice(len(types), p=w)]
        sr = generate_slice_request(request_id=f"m_{i:04d}", substrate=sub, rng=rng, slice_type=st)
        c = classify_slice(sr, sub)
        for k, v in c.items():
            agg[k] += int(v)
    return agg


def main():
    sub = R.generate_family_instance({f.short_name: f for f in R.ALL_FAMILIES}[FAMILY], seed=0)
    print(f"Shift diagnostic on {FAMILY} (n={N_PER}/type)\n")

    print("=== per-slice-type (which types force partitioning?) ===")
    print(f"{'type':8s} {'coloc_feas':>11s} {'partition_NEEDED':>17s}")
    pn = {}
    for i, st in enumerate(SliceType):
        c = classify_stream(sub, st, N_PER, seed=100 + i)
        t = N_PER
        pn[st] = 100 * c["partition_needed"] / t
        print(f"{st.value:8s} {100*c['coloc_feasible']/t:10.1f}% {100*c['partition_needed']/t:16.1f}%")

    # Build candidate mixes from the data: partition-heavy favors high-partition_needed
    # types; coloc-heavy favors low ones. Baseline = production weights.
    ranked = sorted(SliceType, key=lambda s: -pn[s])
    part_heavy = {ranked[0]: 0.40, ranked[1]: 0.30, ranked[2]: 0.15, ranked[3]: 0.10, ranked[4]: 0.05}
    coloc_heavy = {ranked[4]: 0.40, ranked[3]: 0.30, ranked[2]: 0.15, ranked[1]: 0.10, ranked[0]: 0.05}
    from orion.sim.slice_generator import _SLICE_TYPE_WEIGHTS as BASE

    print("\n=== candidate segment mixes (end-to-end partition_needed) ===")
    for name, mix in [("baseline (production)", BASE),
                      ("partition-heavy", part_heavy),
                      ("coloc-heavy", coloc_heavy)]:
        t = 200
        c = classify_mix(sub, mix, t, seed=7)
        wl = " ".join(f"{k.value}={v:.2f}" for k, v in mix.items())
        print(f"  {name:22s} partition_needed={100*c['partition_needed']/t:5.1f}%  coloc_feas={100*c['coloc_feasible']/t:5.1f}%")
        print(f"      weights: {wl}")

    print("\nSchedule candidates for the pre-reg draft:")
    print("  A. MIX shift: baseline -> partition-heavy at a mid-stream change point.")
    print("     Optimum moves (more chains must partition). Verified above if partition-heavy")
    print("     partition_needed >> baseline.")
    print("  B. LOAD-SATURATION shift (needs ONLINE cumulative eval, not this stateless probe):")
    print("     mid-stream lambda spike saturates the colocation tiers so slices that")
    print("     colocated pre-shift must partition post-shift. Verify online before ratifying.")


if __name__ == "__main__":
    main()
