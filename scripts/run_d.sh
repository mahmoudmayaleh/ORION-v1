#!/usr/bin/env bash
# Track D — the 26.7 h run. Detached, survives session close.
# Resumable: d_runner banks each cell to data/d_cells/ and skips banked cells,
# so re-running this script after any kill/crash continues where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs logs data
S=runs/track_d_status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$S"; }

rm -f runs/TRACK_D_COMPLETE
mark "TRACK D START — 300 arrivals, seeds 42/43/44, n=3 on LLM-path arms"
mark "  order: Plain -> follow_prior x3 -> RL-alone -> Full-ORION x3 (trained arms LAST)"
mark "  est 26.7 h (calibration-measured); local only, no API cost"
mark "  banked cells: $(ls data/d_cells 2>/dev/null | wc -l) already present (skipped on resume)"

if .venv/bin/python scripts/d_runner.py --tag D > logs/track_d.log 2>&1; then
  mark "TRACK D DONE"
  sed -n '/TRACK D — 300-arrival/,$p' logs/track_d.log | tee -a "$S"
else
  mark "TRACK D FAILED (exit $?) — banked cells are intact; re-run this script to resume"
  tail -20 logs/track_d.log | tee -a "$S"
fi
mark "cells banked: $(ls data/d_cells 2>/dev/null | wc -l)"
touch runs/TRACK_D_COMPLETE
