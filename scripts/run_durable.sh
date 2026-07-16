#!/usr/bin/env bash
# Durable 3-arm experiment supervisor for ORION.
#  - Ensures BOTH GPU LLM servers (A6000): :8000 for Memory-off, :8001 for Full-M^B.
#    The two LLM arms run concurrently, one server each (the Python server
#    serializes requests, so one server cannot overlap two arms).
#  - Runs five_arm_runner.py with auto-restart; resumes from per-seed checkpoints.
#  - Detached in `screen`; survives SSH drop / laptop close. With the cron watchdog
#    + @reboot entry, also survives session kill and full reboot.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p runs data/checkpoints
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runs/three_arm_${STAMP}.log"
ln -sf "$(basename "$LOG")" runs/latest.log

log(){ echo "[supervisor $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure_server(){
  local port="$1"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then return 0; fi
  log "LLM server :${port} down -> starting on A6000"
  setsid ./scripts/start_llm_gpu.sh "$port" > "llm_server_${port}.log" 2>&1 < /dev/null &
  for i in $(seq 1 80); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { log "LLM server :${port} healthy"; return 0; }
    sleep 3
  done
  log "LLM server :${port} failed to come up within timeout"; return 1
}

ensure_all_servers(){
  ensure_server 8000 && ensure_server 8001
}

log "=== durable run start (log: $LOG) ==="
ensure_all_servers || { log "aborting: LLM servers not both healthy"; exit 1; }

MAX=10; n=0
while true; do
  log "launching runner (attempt $((n+1)) of $MAX)"
  # nice -19: the ceiling phase is CPU-heavy; yield to interactive desktop use on
  # this shared box so a human is less likely to kill it.
  nice -n 19 ./.venv/bin/python scripts/five_arm_runner.py 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    log "=== EXPERIMENT COMPLETE (rc=0) ==="
    touch runs/COMPLETE
    break
  fi
  n=$((n+1))
  if [ "$n" -ge "$MAX" ]; then
    log "GIVING UP after $MAX failed attempts (last rc=$rc). Inspect $LOG."
    break
  fi
  log "runner exited rc=$rc; restart #$n in 15s (resumes from per-seed checkpoints)"
  sleep 15
  ensure_all_servers || log "warning: an LLM server is still down; retrying runner anyway"
done
log "supervisor exiting"
