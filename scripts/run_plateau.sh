#!/usr/bin/env bash
# Durable MDO plateau check on C+_T-_B- (CPU, no LLM). Reruns on failure until done.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results/wp7
[ -f runs/MDO_PLATEAU_DONE ] && { echo "already done"; exit 0; }
LOG="runs/mdo_plateau_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" runs/mdo_plateau_latest.log
n=0; MAX=6
while [ ! -f runs/MDO_PLATEAU_DONE ] && [ $n -lt $MAX ]; do
  echo "[plateau $(date +%H:%M:%S)] attempt $((n+1))/$MAX" | tee -a "$LOG"
  nice -n 19 ./.venv/bin/python -u scripts/mdo_plateau_check.py \
    --family C+_T-_B- --seed 42 --rounds 80 --arrivals 60 2>&1 | tee -a "$LOG"
  [ -f runs/MDO_PLATEAU_DONE ] && break
  n=$((n+1)); sleep 10
done
echo "[plateau $(date +%H:%M:%S)] exiting (done=$([ -f runs/MDO_PLATEAU_DONE ] && echo yes || echo no))" | tee -a "$LOG"
