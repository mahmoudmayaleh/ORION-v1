#!/usr/bin/env bash
# Recover the chat template that reproduces R.2's ~84 admits (86.6%). The July-8 server that
# produced R.2 used "fallback chat format: llama-2" (per runs/llm_8000.log line 767). Sweep
# candidate --chat_format values on a test port (:8003), 30-arrival anchor each; deploy the first
# that reproduces (>=65% admit) to :8000/1/2 and update start_llm_gpu.sh. Else flag TEMPLATE_FAILED.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs/recover_template.log
mkdir -p runs
: > "$LOG"
echo "[recover $(date '+%F %T')] start" | tee -a "$LOG"

FORMATS=("llama-2" "chatml" "mistral-instruct" "alpaca" "vicuna" "zephyr")
A6000=$(nvidia-smi -L | grep A6000 | grep -oE 'GPU-[0-9a-f-]+')
CUDART=$(dirname "$(find .venv/lib/python3.11/site-packages/nvidia -name libcudart.so.12 | head -1)")
CUBLAS=$(dirname "$(find .venv/lib/python3.11/site-packages/nvidia -name libcublas.so.12 | head -1)")

launch_test() {  # $1=format, on :8003
  screen -S llmtest -X quit 2>/dev/null; sleep 2
  screen -dmS llmtest bash -lc "cd $(pwd) && CUDA_VISIBLE_DEVICES=$A6000 LD_LIBRARY_PATH=$CUDART:$CUBLAS .venv/bin/python -m llama_cpp.server --model models/LLama-3-8B-Tele-it.Q4_K_M.gguf --n_gpu_layers -1 --n_ctx 8192 --chat_format $1 --host 127.0.0.1 --port 8003 >> runs/llm_8003_test.log 2>&1"
  sleep 60
}

WINNER=""
for fmt in "${FORMATS[@]}"; do
  echo "[recover] testing --chat_format $fmt" | tee -a "$LOG"
  launch_test "$fmt"
  rm -f /tmp/orion_local_llm.lock
  OUT=$(.venv/bin/python quick_anchor.py 30 8003 2>>"$LOG")
  echo "$OUT" | tee -a "$LOG"
  RATE=$(echo "$OUT" | grep -oE '\([0-9]+%\)' | grep -oE '[0-9]+' | head -1)
  echo "[recover] $fmt -> rate=${RATE:-NA}%" | tee -a "$LOG"
  if [ -n "${RATE:-}" ] && [ "$RATE" -ge 65 ]; then WINNER="$fmt"; break; fi
done
screen -S llmtest -X quit 2>/dev/null; sleep 2

if [ -n "$WINNER" ]; then
  echo "[recover] WINNER=$WINNER -> deploying to :8000/1/2" | tee -a "$LOG"
  sed -i "s/--chat_format [^ ]*/--chat_format $WINNER/" scripts/start_llm_gpu.sh
  for p in 8000 8001 8002; do screen -S llm$p -X quit 2>/dev/null; done; sleep 3
  for p in 8000 8001 8002; do
    screen -dmS llm$p bash -lc "cd $(pwd) && ./scripts/start_llm_gpu.sh $p >> runs/llm_$p.log 2>&1"
  done
  sleep 65
  rm -f /tmp/orion_local_llm.lock
  echo "$WINNER" > runs/TEMPLATE_RECOVERED
  rm -f runs/TEMPLATE_FAILED
  echo "[recover $(date '+%F %T')] RECOVERED format=$WINNER, servers redeployed" | tee -a "$LOG"
else
  echo "no format reproduced R.2 (all <65% on 30-arrival anchor)" > runs/TEMPLATE_FAILED
  rm -f runs/TEMPLATE_RECOVERED
  echo "[recover $(date '+%F %T')] FAILED — no candidate reproduced R.2" | tee -a "$LOG"
fi
