#!/usr/bin/env bash
# Cron */2: keep the colocation-prior gate (and its Agent B LLM server) alive until done.
cd "$(dirname "$0")/.." || exit 0
[ -f runs/GATE_COLOCATION_DONE ] && exit 0
# 1. Ensure the Agent B LLM server on :8000 (the gate needs it for its whole run).
if ! curl -s -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "[gate-watchdog $(date '+%F %T')] LLM :8000 down -> restarting" >> runs/gate_watchdog.log
  setsid ./scripts/start_llm_gpu.sh 8000 > llm_server_8000.log 2>&1 < /dev/null &
  sleep 25
fi
# 2. Ensure the gate session.
screen -ls 2>/dev/null | grep -qE '[0-9]+\.gate' && exit 0
screen -dmS gate ./scripts/run_gate.sh
