#!/usr/bin/env bash
# Durable AMENDED colocation-prior gate (h^m obs enrichment + entropy-floor schedule +
# inline cost profiling). PREREG 2026-07-11 §M. LLM arms need llama.cpp on :8000. New --tag
# so it CANNOT resume the 2026-07-10 pre-enrichment cells. Resumes its own cells on restart.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results/wp7
MARKER=runs/GATE_COLOCATION_AMENDED_DONE
[ -f "$MARKER" ] && { echo "already done"; exit 0; }
LOG="runs/gate_amended_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" runs/gate_amended_latest.log
n=0; MAX=12
while [ ! -f "$MARKER" ] && [ $n -lt $MAX ]; do
  # arms 2 & 3 need the LLM server; keep it alive across reruns.
  if ! curl -s -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[amended $(date +%H:%M:%S)] LLM :8000 down -> starting" | tee -a "$LOG"
    setsid ./scripts/start_llm_gpu.sh 8000 > llm_server_8000.log 2>&1 < /dev/null &
    sleep 30
  fi
  echo "[amended $(date +%H:%M:%S)] attempt $((n+1))/$MAX -> $LOG" | tee -a "$LOG"
  nice -n 19 ./.venv/bin/python -u scripts/gate_colocation_prior.py \
    --family C+_T-_B- --seeds 42 43 44 --rounds 15 --arrivals 60 \
    --bc-scenarios 2000 --bc-epochs 6 --port 8000 \
    --tag amended --ent-c0 0.03 --ent-floor 0.01 2>&1 | tee -a "$LOG"
  [ -f "$MARKER" ] && break
  n=$((n+1)); sleep 15
done
echo "[amended $(date +%H:%M:%S)] exiting (done=$([ -f "$MARKER" ] && echo yes || echo no))" | tee -a "$LOG"
