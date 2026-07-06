#!/bin/bash
# Item 5: LLM backend setup for the GPU server (T400 4GB)
#
# Serves LLaMA-3-8B-Instruct quantized (4-bit GPTQ/AWQ) via vLLM
# on an OpenAI-compatible endpoint at localhost:8000.
#
# Prerequisites:
#   pip install vllm
#   or: pip install llama-cpp-python[server]
#
# REQUIRED PYTHON ENV FOR THE FIVE-ARM RUN (do NOT omit the retrieval extra):
#   uv pip install -e ".[actors,retrieval]"
# The [retrieval] extra pins rank-bm25, which M^B retrieval needs. Without it,
# retrieval silently degrades to recency-only (build_index now hard-fails to
# make this impossible to miss on a fresh venv / new hardware).
#
# The five_arm_runner.py expects LLMConfig(base_url="http://localhost:8000/v1")

set -e

MODEL_ID="TheBloke/Meta-Llama-3-8B-Instruct-GPTQ"
# Alternative for llama-cpp: "bartowski/Meta-Llama-3-8B-Instruct-GGUF"

echo "================================================================"
echo "LLM Server Setup for ORION Five-Arm Experiment"
echo "================================================================"
echo ""
echo "Option A: vLLM (recommended if GPU VRAM >= 6GB with quantization)"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model $MODEL_ID \\"
echo "    --quantization gptq \\"
echo "    --max-model-len 4096 \\"
echo "    --gpu-memory-utilization 0.90 \\"
echo "    --port 8000"
echo ""
echo "Option B: llama-cpp-python (for smaller GPUs, GGUF quantization)"
echo "  python -m llama_cpp.server \\"
echo "    --model bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \\"
echo "    --n_gpu_layers -1 \\"
echo "    --n_ctx 4096 \\"
echo "    --port 8000"
echo ""
echo "Option C: Ollama (simplest setup)"
echo "  ollama serve &"
echo "  ollama pull llama3:8b-instruct-q4_K_M"
echo "  # Set base_url to http://localhost:11434/v1 in LLMConfig"
echo ""
echo "After starting the server, test with:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "Then launch the experiment:"
echo "  cd ~/ORION"
echo "  python scripts/five_arm_runner.py --seed 42"
echo "================================================================"
