#!/usr/bin/env bash
# Durable load-concentration probe. RL-alone only (no LLM server needed). Reruns on
# transient failure until the DONE marker lands. Resumes via the per-seed checkpoint.
set -u
cd "$(dirname "$0")/.."
MARKER="runs/LOAD_CONC_DONE"
LOG="runs/loadconc_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" runs/loadconc_latest.log
MAX=6
i=0
while [ ! -f "$MARKER" ] && [ "$i" -lt "$MAX" ]; do
  i=$((i+1))
  echo "[run_loadconc] attempt $i/$MAX -> $LOG" | tee -a "$LOG"
  nice -n 19 ./.venv/bin/python -u scripts/load_concentration_probe.py \
    --family C+_T-_B- --seeds 42 43 44 --rounds 15 --arrivals 60 \
    --bc-scenarios 2000 --bc-epochs 6 >> "$LOG" 2>&1
  [ -f "$MARKER" ] && break
  echo "[run_loadconc] exited without marker, retrying in 10s" | tee -a "$LOG"
  sleep 10
done
echo "[run_loadconc] done (marker=$([ -f "$MARKER" ] && echo yes || echo NO))" | tee -a "$LOG"
