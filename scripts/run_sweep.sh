#!/usr/bin/env bash
# Durable MDO reward-rebalance sweep on C+_T-_B- (CPU, no LLM). Resumes on restart.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results/wp7
[ -f runs/MDO_REWARD_SWEEP_DONE ] && { echo "already done"; exit 0; }
LOG="runs/mdo_reward_sweep_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" runs/mdo_reward_sweep_latest.log
n=0; MAX=10
while [ ! -f runs/MDO_REWARD_SWEEP_DONE ] && [ $n -lt $MAX ]; do
  echo "[sweep $(date +%H:%M:%S)] attempt $((n+1))/$MAX" | tee -a "$LOG"
  nice -n 19 ./.venv/bin/python -u scripts/mdo_reward_sweep.py \
    --family C+_T-_B- --seeds 42 43 44 --rounds 80 --arrivals 60 2>&1 | tee -a "$LOG"
  [ -f runs/MDO_REWARD_SWEEP_DONE ] && break
  n=$((n+1)); sleep 10
done
echo "[sweep $(date +%H:%M:%S)] exiting (done=$([ -f runs/MDO_REWARD_SWEEP_DONE ] && echo yes || echo no))" | tee -a "$LOG"
