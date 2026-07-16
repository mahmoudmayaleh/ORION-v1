#!/usr/bin/env bash
# §O regression ladder — one command, all-green writes runs/LADDER_O_GREEN,
# which scripts/run_gate_O.sh requires before it will start.
#
# Run this ON THE BOX (cri-pt-7865) before the gate: the ladder must be green
# in the environment the gate actually runs in, not only on the dev machine.
# (Local pre-verification 2026-07-13, Windows/torch 2.6: all six green.)
#
# Budgets (measured, PREREG_AMENDMENT_2026-07-13_O.md):
#   L1 60 rounds — §O critic transient lasts ~20 rounds; converged+stable by ~R50
#   L2 100 rounds — switching task converges ~R44, held through R100
#   L3 60 rounds  — trap avoidance + high admission
#   L6 10 rounds  — KL frame behavioral check (beta=5)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
FAIL=0

run() {
  echo "== $1 =="
  shift
  "$@" || FAIL=1
}

run "L1 constant-best (bar: >95% sel, no walk-away, EV>=0.5 tail5)" \
    $PY scripts/audit/canary1_constant_best.py 60
run "L2 observation-conditional (bar: tail-3 tracking >90%)" \
    $PY scripts/audit/canary2_obs_conditional.py 100
run "L3 saturation (bar: late-A <30%, admit >=80%)" \
    $PY scripts/audit/canary3_saturation.py 60
run "L4 phase-2 asserts (2.3/2.5/2.6/2.8+2.1)" \
    $PY scripts/audit/test_phase2_asserts.py
run "L5 update-direction" \
    $PY scripts/audit/test_update_direction.py
run "L6 KL-frame behavioral (bar: tail-3 agreement >=0.35, KL decreasing, 0 skips)" \
    $PY scripts/audit/canary_l6_kl_frame.py 10

# L1-L3/L6 write pass booleans into their result JSONs; verify them explicitly
# (their scripts exit 0 even on FAIL — the JSON is the record).
$PY - <<'EOF' || FAIL=1
import json, sys
from pathlib import Path
out = Path("scripts/audit/out")
checks = {
    "L1": out / "canary1_result.json",
    "L2": out / "canary2_result.json",
    "L3": out / "canary3_result.json",
    "L6": out / "canary_l6_result.json",
}
bad = [k for k, p in checks.items()
       if not p.exists() or not json.load(open(p)).get("pass")]
if bad:
    print(f"LADDER NOT GREEN: {bad}")
    sys.exit(1)
print("ladder JSON verdicts: all pass")
EOF

if [ "$FAIL" -eq 0 ]; then
  mkdir -p runs
  date -u +"%Y-%m-%dT%H:%M:%SZ" > runs/LADDER_O_GREEN
  echo "ALL GREEN -> runs/LADDER_O_GREEN written; gate may fire."
else
  rm -f runs/LADDER_O_GREEN
  echo "LADDER NOT GREEN — gate launcher will refuse to start."
  exit 1
fi
