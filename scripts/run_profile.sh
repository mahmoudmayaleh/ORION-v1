#!/usr/bin/env bash
# Run the per-decision profiler on an idle A6000 (single server) after the main
# batch run completes. Writes results/ + touches runs/PROFILE_DONE when finished.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results runs

# NVML python binding for GPU energy sampling (userspace, no deps).
./.venv/bin/python -c "import pynvml" 2>/dev/null || \
  ./.venv/bin/python -m pip install --quiet nvidia-ml-py 2>/dev/null || true

# Exclusive-ish GPU: keep ONE server (:8000), stop the 2nd (:8001) so only one
# model is resident (clean VRAM baseline, no stray traffic). Main run is done.
pkill -f "llama_cpp.server.*--port 8001" 2>/dev/null || true
sleep 3

# Ensure :8000 up.
if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  setsid ./scripts/start_llm_gpu.sh 8000 > llm_server_8000.log 2>&1 < /dev/null &
  for i in $(seq 1 80); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    sleep 3
  done
fi

echo "[profile $(date '+%F %T')] starting profiler"
./.venv/bin/python scripts/profile_decision.py --port 8000 --out results 2>&1 | tee runs/profile.log
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ]; then
  touch runs/PROFILE_DONE
  echo "[profile $(date '+%F %T')] DONE -> results/profile_summary.md"
else
  echo "[profile $(date '+%F %T')] FAILED rc=$rc (will retry on next watchdog tick)"
fi
