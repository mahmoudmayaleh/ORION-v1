#!/usr/bin/env bash
# Detached sequence, survives session close: calibration -> B-tele x3 -> C-tele x3.
# All local, no API, no cost. Serialized: the local llama.cpp server is single-slot
# and the LLM lock in llm_backend.py refuses concurrent jobs (the 2026-07-15 lesson).
# n=3 per EXPERIMENT_PROTOCOL Amendment 9 (LLM-path cells, median+range).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs logs data
S=runs/calib_bc_status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$S"; }

: > "$S"
rm -f runs/CALIB_BC_COMPLETE
mark "SEQUENCE START (calibration -> B-tele x3 -> C-tele x3) — local only, no API"

# ---- 1. Track D wall-clock calibration (gates §T commit) ----
mark "STEP 1/3 D calibration: one 300-arrival follow_prior mb=None cell (~58 min est)"
if .venv/bin/python d_calibration.py > logs/d_calibration.log 2>&1; then
  mark "CALIBRATION DONE"
  grep -E "WALL|per LLM call|linearity|CORE|threshold" logs/d_calibration.log | tee -a "$S"
else
  mark "CALIBRATION FAIL (see logs/d_calibration.log)"
  tail -5 logs/d_calibration.log | tee -a "$S"
fi

# ---- 2. Track B tele, n=3 ----
for r in 1 2 3; do
  mark "STEP 2/3 Track B tele rep $r/3 (100 intents x K^A on/off, ~38 min est)"
  if .venv/bin/python scripts/track_b_runner.py --arms tele --rep "$r" \
       > "logs/track_b_tele_rep$r.log" 2>&1; then
    mark "  B rep $r DONE"
  else
    mark "  B rep $r FAIL (see logs/track_b_tele_rep$r.log)"
    tail -3 "logs/track_b_tele_rep$r.log" | tee -a "$S"
  fi
done

# ---- 3. Track C tele, n=3 ----
for r in 1 2 3; do
  mark "STEP 3/3 Track C tele rep $r/3 (30 RC arrivals, ~6 min est)"
  if .venv/bin/python scripts/track_c_runner.py --arms tele --rep "$r" \
       > "logs/track_c_tele_rep$r.log" 2>&1; then
    mark "  C rep $r DONE"
  else
    mark "  C rep $r FAIL (see logs/track_c_tele_rep$r.log)"
    tail -3 "logs/track_c_tele_rep$r.log" | tee -a "$S"
  fi
done

mark "SEQUENCE DONE — read: data/d_calibration.json, data/track_b_results_tele_full_rep{1,2,3}.json,"
mark "                      data/track_c_results_rep{1,2,3}.json"
touch runs/CALIB_BC_COMPLETE
