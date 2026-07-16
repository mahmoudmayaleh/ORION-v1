#!/usr/bin/env python3
"""Item 6: Analysis script for the three-arm learning-curve experiment.

Reads data/three_arm_results.json and produces:
  1. Per-family FoC bar charts (extrapolation + interpolation)
  2. Paired differences (Full-M^B minus best static) with spread
  3. Reject-reason breakdown per arm per family
  4. Automatic pass/fail against pre-registered success criteria
  5. Summary table for the paper

Designed to run against mock data for verification, then against real data
without changes. The figure is ready hours after the run lands.

Usage:
  python scripts/analyze_results.py                        # default file
  python scripts/analyze_results.py --input path/to/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Protocol constants ──────────────────────────────────────────────────────

B_MINUS_PRIMARY = ["C-_T+_B-", "C+_T-_B-"]
CONTROL_FAMILY = "C-_T+_B+"
ARM_ORDER = ["RA-ColocFB", "Memory-off", "Full-M^B"]
STATIC_ARMS = ["RA-ColocFB"]
LLM_ARMS = ["Memory-off", "Full-M^B"]


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    # §P tagged format wraps the record list with provenance; older runs are a
    # bare list. Accept both.
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def group_by(records, *keys):
    """Group records by one or more keys."""
    groups = defaultdict(list)
    for r in records:
        key = tuple(r[k] for k in keys)
        groups[key].append(r)
    return groups


def mean_foc(records):
    focs = [r["foc"] for r in records]
    return np.mean(focs) if focs else 0.0


def std_foc(records):
    focs = [r["foc"] for r in records]
    return np.std(focs) if len(focs) > 1 else 0.0


def analyze(results: list[dict]):
    """Full analysis pipeline."""

    print("=" * 90)
    print("THREE-ARM EXPERIMENT ANALYSIS")
    print("=" * 90)

    seeds = sorted(set(r.get("seed", 0) for r in results))
    n_seeds = max(len(seeds), 1)

    # ── 1. Per-family FoC summary ───────────────────────────────────────────

    for phase_label, phase_name in [("extrap", "EXTRAPOLATION (held-out families)"),
                                     ("interp", "INTERPOLATION (unseen instances)")]:
        phase_records = [r for r in results if r["phase"] == phase_label]
        if not phase_records:
            continue

        print(f"\n{'─' * 90}")
        print(f"  {phase_name}")
        print(f"{'─' * 90}")

        families = sorted(set(r["family"] for r in phase_records))

        # Header
        header = f"  {'Family':<14}"
        for arm in ARM_ORDER:
            header += f"  {arm:>14}"
        print(header)
        print("  " + "─" * (14 + 16 * len(ARM_ORDER)))

        for family in families:
            row = f"  {family:<14}"
            for arm in ARM_ORDER:
                arm_records = [r for r in phase_records
                              if r["family"] == family and r["arm"] == arm]
                if arm_records:
                    m = 100 * mean_foc(arm_records)
                    s = 100 * std_foc(arm_records)
                    if s > 0.1:
                        row += f"  {m:>10.1f}±{s:<3.1f}"
                    else:
                        row += f"  {m:>13.1f}%"
                else:
                    row += f"  {'—':>14}"
            print(row)

    # ── 2. Paired differences ───────────────────────────────────────────────

    print(f"\n{'─' * 90}")
    print("  PAIRED DIFFERENCES (Full-M^B minus best static)")
    print(f"{'─' * 90}")

    for phase_label, phase_name in [("extrap", "Extrapolation"), ("interp", "Interpolation")]:
        phase_records = [r for r in results if r["phase"] == phase_label]
        if not phase_records:
            continue

        print(f"\n  {phase_name}:")
        families = sorted(set(r["family"] for r in phase_records))

        for family in families:
            full_focs = [r["foc"] for r in phase_records
                        if r["family"] == family and r["arm"] == "Full-M^B"]
            best_static_focs = []
            for arm in STATIC_ARMS:
                arm_focs = [r["foc"] for r in phase_records
                           if r["family"] == family and r["arm"] == arm]
                if arm_focs:
                    best_static_focs.append(np.mean(arm_focs))

            if full_focs and best_static_focs:
                full_mean = 100 * np.mean(full_focs)
                best_static = 100 * max(best_static_focs)
                delta = full_mean - best_static
                spread = 100 * np.std(full_focs) if len(full_focs) > 1 else 0
                marker = "✓" if delta > spread and delta > 0 else "✗"
                print(f"    {family:<14}  Full={full_mean:5.1f}%  Static={best_static:5.1f}%  "
                      f"Δ={delta:+5.1f} pp  spread={spread:.1f}  {marker}")

    # ── 3. Reject-reason breakdown ──────────────────────────────────────────

    print(f"\n{'─' * 90}")
    print("  REJECT-REASON BREAKDOWN (extrapolation)")
    print(f"{'─' * 90}")

    extrap = [r for r in results if r["phase"] == "extrap"]
    families = sorted(set(r["family"] for r in extrap))

    for family in families:
        print(f"\n  {family}:")
        for arm in ARM_ORDER:
            arm_records = [r for r in extrap
                          if r["family"] == family and r["arm"] == arm]
            if not arm_records:
                continue
            # Aggregate reject reasons
            agg_reasons = defaultdict(int)
            for r in arm_records:
                for reason, count in r.get("reject_reasons", {}).items():
                    agg_reasons[reason] += count
            if agg_reasons:
                reason_str = ", ".join(f"{k}:{v}" for k, v in
                                       sorted(agg_reasons.items(), key=lambda x: -x[1]))
                print(f"    {arm:<14}: {reason_str}")

    # ── 4. Pre-registered success criteria ──────────────────────────────────

    print(f"\n{'=' * 90}")
    print("  SUCCESS CRITERIA CHECK (pre-registered)")
    print(f"{'=' * 90}")

    extrap = [r for r in results if r["phase"] == "extrap"]

    # Primary: Full-M^B vs best static on B- test families
    print("\n  PRIMARY: Full-M^B vs best static on B- families")
    primary_pass = True
    for family in B_MINUS_PRIMARY:
        full_focs = [r["foc"] for r in extrap
                    if r["family"] == family and r["arm"] == "Full-M^B"]
        best_static = 0
        for arm in STATIC_ARMS:
            arm_focs = [r["foc"] for r in extrap
                       if r["family"] == family and r["arm"] == arm]
            if arm_focs:
                best_static = max(best_static, np.mean(arm_focs))

        if full_focs:
            full_mean = np.mean(full_focs)
            full_std = np.std(full_focs) if len(full_focs) > 1 else 0
            delta = full_mean - best_static
            clears_spread = delta > full_std
            status = "PASS" if delta > 0 and clears_spread else "FAIL"
            if delta <= 0:
                primary_pass = False
            print(f"    {family}: Full={100*full_mean:.1f}% Static={100*best_static:.1f}% "
                  f"Δ={100*delta:+.1f}pp spread={100*full_std:.1f}% → {status}")
        else:
            print(f"    {family}: NO DATA")
            primary_pass = False

    # No-regression control
    print(f"\n  NO-REGRESSION: {CONTROL_FAMILY}")
    control = [r for r in extrap if r["family"] == CONTROL_FAMILY]
    if control:
        for arm in ARM_ORDER:
            arm_focs = [r["foc"] for r in control if r["arm"] == arm]
            if arm_focs:
                m = 100 * np.mean(arm_focs)
                print(f"    {arm:<14}: {m:.1f}%", end="")
                if arm in LLM_ARMS:
                    static_focs = [r["foc"] for r in control if r["arm"] == "RA-ColocFB"]
                    if static_focs:
                        static_m = 100 * np.mean(static_focs)
                        gap = m - static_m
                        status = "OK" if abs(gap) <= 1.5 else "REGRESSION"
                        print(f"  (Δ vs static: {gap:+.1f}pp → {status})", end="")
                print()

    # Secondary: Full vs Memory-off
    print("\n  SECONDARY:")
    for phase_label in ["extrap", "interp"]:
        phase_r = [r for r in results if r["phase"] == phase_label]
        if not phase_r:
            continue
        for a, b in [("Full-M^B", "Memory-off")]:
            a_focs = [r["foc"] for r in phase_r if r["arm"] == a]
            b_focs = [r["foc"] for r in phase_r if r["arm"] == b]
            if a_focs and b_focs:
                delta = 100 * (np.mean(a_focs) - np.mean(b_focs))
                status = "PASS" if delta > 0.5 else "FAIL" if delta < -0.5 else "TIE"
                print(f"    {phase_label}: {a}={100*np.mean(a_focs):.1f}% vs "
                      f"{b}={100*np.mean(b_focs):.1f}% Δ={delta:+.1f}pp → {status}")

    # ── 5. Pre-named outcome classification ─────────────────────────────────

    print(f"\n{'─' * 90}")
    print("  OUTCOME CLASSIFICATION")
    print(f"{'─' * 90}")

    # Check gain locations
    extrap_gain = False
    interp_gain = False

    for phase_label, flag_name in [("extrap", "extrap_gain"), ("interp", "interp_gain")]:
        phase_r = [r for r in results if r["phase"] == phase_label]
        full_focs = [r["foc"] for r in phase_r if r["arm"] == "Full-M^B"]
        static_focs = []
        for arm in STATIC_ARMS:
            static_focs.extend([r["foc"] for r in phase_r if r["arm"] == arm])
        if full_focs and static_focs:
            if np.mean(full_focs) > np.mean(static_focs):
                if flag_name == "extrap_gain":
                    extrap_gain = True
                else:
                    interp_gain = True

    memoff_matches = True
    all_phase_r = [r for r in results if r["phase"] in ("extrap", "interp")]
    full_all = [r["foc"] for r in all_phase_r if r["arm"] == "Full-M^B"]
    memoff_all = [r["foc"] for r in all_phase_r if r["arm"] == "Memory-off"]
    if full_all and memoff_all:
        memoff_matches = abs(np.mean(full_all) - np.mean(memoff_all)) < 0.01

    if extrap_gain and interp_gain:
        print("  → SUCCESS: Memory adapts across topology families")
    elif interp_gain and not extrap_gain:
        print("  → MEMORIZATION: Adapts to seen families, not unseen combinations")
    elif extrap_gain and not interp_gain:
        print("  → NARROW: Memory helps on unseen combinations only")
    elif not extrap_gain and not interp_gain:
        print("  → NO GAIN: Plan layer does not need episodic memory (negative result)")

    if memoff_matches:
        print("  → Memory-off matches Full-M^B: LLM capability, not memory, drives result")
    else:
        print("  → Memory-off differs from Full-M^B: episodic retrieval matters")

    print(f"\n{'=' * 90}")

    # ── 6. Paper-ready summary table ────────────────────────────────────────

    print("\n  PAPER TABLE (fraction-of-ceiling, %)")
    print(f"{'─' * 90}")

    extrap = [r for r in results if r["phase"] == "extrap"]
    if extrap:
        families = sorted(set(r["family"] for r in extrap))
        header = f"  {'Family':<14}  {'Role':<12}"
        for arm in ARM_ORDER:
            short = arm.replace("ColocFB", "CFB").replace("Memory-off", "Mem-off")
            header += f"  {short:>10}"
        print(header)
        print("  " + "─" * 80)

        for family in families:
            if family in B_MINUS_PRIMARY:
                role = "Primary"
            elif family == CONTROL_FAMILY:
                role = "Control"
            else:
                role = "—"
            row = f"  {family:<14}  {role:<12}"
            for arm in ARM_ORDER:
                arm_records = [r for r in extrap
                              if r["family"] == family and r["arm"] == arm]
                if arm_records:
                    m = 100 * mean_foc(arm_records)
                    row += f"  {m:>9.1f}%"
                else:
                    row += f"  {'—':>10}"
            print(row)

    print(f"\n{'=' * 90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/three_arm_results.json",
                        help="Path to results JSON")
    args = parser.parse_args()

    results = load_results(args.input)
    analyze(results)


if __name__ == "__main__":
    main()
