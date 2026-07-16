#!/usr/bin/env bash
# Durable WP7 reduced-scale on-claim run (3 arms, C-_T-_B-, seed 42).
# Sequential arms, one llama server on :8000. Resumes via per-arm checkpoints.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results/wp7
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runs/wp7_${STAMP}.log"
ln -sf "$(basename "$LOG")" runs/wp7_latest.log
log(){ echo "[wp7-sup $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure_server(){
  local port="$1"
  curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
  log "LLM server :${port} down -> starting on A6000"
  setsid ./scripts/start_llm_gpu.sh "$port" > "llm_server_${port}.log" 2>&1 < /dev/null &
  for i in $(seq 1 80); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { log "server :${port} up"; return 0; }
    sleep 3
  done
  log "server :${port} failed to come up"; return 1
}

log "=== WP7 durable run start (log: $LOG) ==="
ensure_server 8000 || { log "abort: server not up"; exit 1; }

MAX=10; n=0
while true; do
  log "launch wp7_runner (attempt $((n+1))/$MAX)"
  nice -n 19 ./.venv/bin/python -u scripts/wp7_runner.py \
    --family C-_T-_B- --seed 42 --rounds 30 --arrivals 60 \
    --arms RL-alone "LLM+RL-memoff" "LLM+RL-full" --port 8000 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    log "=== WP7 COMPLETE (rc=0) ==="; touch runs/WP7_COMPLETE; break
  fi
  n=$((n+1))
  if [ "$n" -ge "$MAX" ]; then log "GIVING UP after $MAX (last rc=$rc)"; break; fi
  log "rc=$rc; restart #$n in 15s (resumes via per-arm checkpoints)"
  sleep 15
  ensure_server 8000 || log "warn: server still down; retrying anyway"
done
log "wp7 supervisor exiting"
