#!/usr/bin/env bash
# Start a llama.cpp server for Agent B on the RTX A6000 (full GPU offload).
# Usage: start_llm_gpu.sh [PORT]   (default 8000)
# The cu121 llama-cpp-python wheel needs CUDA-12 runtime libs this box lacks
# (no toolkit) -> borrow them from torch 2.5.1+cu121's bundled nvidia pkgs.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8000}"
VENV=./.venv
A6000=$(nvidia-smi -L | grep A6000 | grep -oE 'GPU-[0-9a-f-]+')
CUDART=$(dirname $(find $VENV/lib/python3.11/site-packages/nvidia -name libcudart.so.12 | head -1))
CUBLAS=$(dirname $(find $VENV/lib/python3.11/site-packages/nvidia -name libcublas.so.12 | head -1))
export CUDA_VISIBLE_DEVICES="$A6000"
export LD_LIBRARY_PATH="$CUDART:$CUBLAS:${LD_LIBRARY_PATH:-}"
exec $VENV/bin/python -m llama_cpp.server \
  --model models/LLama-3-8B-Tele-it.Q4_K_M.gguf \
  --n_gpu_layers -1 --n_ctx 8192 --chat_format llama-3 \
  --host 127.0.0.1 --port "$PORT"
