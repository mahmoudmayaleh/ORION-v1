#!/usr/bin/env python
"""Track D wall-clock calibration — one 300-arrival follow_prior mb=None cell.

Firing condition for §T (Δ2-T). The committed D estimate (core 27.1 h) was derived
by decomposing R45's measured 156 min against R.2-prime's 11.5 s/call and assuming
wall-clock is LINEAR in arrivals. This measures that assumption instead of trusting
it: extrapolated budget numbers are exactly what this project spent 2026-07-16
learning not to trust.

Arm is D's arm 3 as re-decided in Δ2-T: cache-OFF, mb=None, the clean plan-quality
baseline. n=1 is sound here -- this is a TIMING calibration, not a number for the
record; its admit count is reported as context only and carries the ±5 serving
noise (EXPERIMENT_PROTOCOL Amendment 9).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import q_pilot_runner as Q
import r_local_runner as R
from orion.provenance import git_provenance, serving_provenance

ARRIVALS = 300
SEED = 42

prov = git_provenance(serving=serving_provenance(8000), tag="D-CALIB",
                      prereg="docs/PREREG_T_2026-07-16.md")
print("provenance: commit=%s dirty=%s prereg_T=%s"
      % (prov["git_commit"][:8], prov["git_dirty"], prov["prereg"]["sha256"][:12]))

Q.ARRIVALS_PER_INSTANCE = ARRIVALS
sub = Q.generate_rc_instance(seed=Q.RC_GEN_SEED, inter_domain_bw_override=R.FROZEN_RC[SEED]["bw"])
agent_b = R._build_local_agent(8000)
kb = R._load_kb()

t0 = time.time()
m = Q.run_q_cell("D-calib", agent_b, kb, None, None, sub, SEED, False)   # mb=None, cache OFF
wall = time.time() - t0

calls = m["total"]
per_call = wall / max(1, calls)

# Re-derive the D estimate from MEASURED 300-arrival cost.
# R45 measured: 156 min for 3 seeds x 2 arms x 250 rounds x 100 arrivals.
# Decomposed there: training ~16.4 min/seed at 100 arrivals -> x3 at 300 = 49.2 min.
train300 = 16.4 * 3
llm300 = wall / 60.0
full_cell = train300 + llm300
core_h = (full_cell * 3 * 3 + train300 * 3 + llm300 * 3 * 3) / 60.0
two_seed_h = (full_cell * 2 * 3 + train300 * 2 + llm300 * 2 * 3) / 60.0

print("\n" + "=" * 74)
print("TRACK D WALL-CLOCK CALIBRATION (measured, 300-arrival follow_prior mb=None)")
print("=" * 74)
print(f"  arrivals            : {calls}")
print(f"  admitted            : {m['admitted']}/{calls}  (context only, ±5 noise)")
print(f"  reasons             : {dict(m.get('reasons', {}))}")
print(f"  WALL                : {wall:.0f}s = {wall/60:.1f} min")
print(f"  per LLM call        : {per_call:.2f}s   (estimate assumed 11.5s)")
print(f"  linearity check     : 100-arrival cell measured ~1150s; 3x = 3450s vs actual {wall:.0f}s"
      f"  -> {'LINEAR' if abs(wall-3450)/3450 < 0.20 else 'NON-LINEAR (estimate invalid)'}")
print()
print("  Re-derived D estimate using this measurement:")
print(f"    Full-ORION/cell   = {train300:.0f} (train) + {llm300:.0f} (llm) = {full_cell:.0f} min")
print(f"    CORE (3 seeds, n=3 on LLM-path arms) = {core_h:.1f} h   [committed estimate: 27.1 h]")
print(f"    fallback (2 seeds, n=3)              = {two_seed_h:.1f} h")
print(f"    threshold 30 h -> {'FITS, fire sequentially' if core_h <= 30 else 'BREAKS: drop seeds before repeats (Amendment 9)'}")

Path("data").mkdir(exist_ok=True)
Path("data/d_calibration.json").write_text(json.dumps({
    "provenance": prov, "arrivals": calls, "admitted": m["admitted"],
    "wall_s": round(wall, 1), "per_call_s": round(per_call, 2),
    "reasons": dict(m.get("reasons", {})),
    "rederived": {"train300_min": train300, "llm300_min": round(llm300, 1),
                  "full_cell_min": round(full_cell, 1),
                  "core_h": round(core_h, 1), "two_seed_h": round(two_seed_h, 1),
                  "committed_estimate_h": 27.1, "threshold_h": 30},
}, indent=2, default=str))
print("\n  -> data/d_calibration.json")
