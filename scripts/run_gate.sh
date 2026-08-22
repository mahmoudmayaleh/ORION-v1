#!/usr/bin/env bash
# Durable colocation-prior gate (LLM arms need llama.cpp on :8000). Resumes on restart.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results/wp7
[ -f runs/GATE_COLOCATION_DONE ] && { echo "already done"; exit 0; }
LOG="runs/gate_colocation_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" runs/gate_colocation_latest.log
n=0; MAX=12
while [ ! -f runs/GATE_COLOCATION_DONE ] && [ $n -lt $MAX ]; do
  echo "[gate $(date +%H:%M:%S)] attempt $((n+1))/$MAX" | tee -a "$LOG"
  nice -n 19 ./.venv/bin/python -u scripts/gate_colocation_prior.py \
    --family C+_T-_B- --seeds 42 43 44 --rounds 15 --arrivals 60 \
    --bc-scenarios 2000 --bc-epochs 6 --port 8000 2>&1 | tee -a "$LOG"
  [ -f runs/GATE_COLOCATION_DONE ] && break
  n=$((n+1)); sleep 15
done
echo "[gate $(date +%H:%M:%S)] exiting (done=$([ -f runs/GATE_COLOCATION_DONE ] && echo yes || echo no))" | tee -a "$LOG"
