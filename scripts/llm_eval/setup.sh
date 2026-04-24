#!/usr/bin/env bash
# One-time setup: builds llama.cpp and installs Python dependencies.
# Run from the repo root: bash scripts/llm_eval/setup.sh

set -e

LLAMA_DIR="$HOME/llama.cpp"

echo "=== [1/3] Installing Python dependencies ==="
pip install llama-cpp-python huggingface_hub

echo "=== [2/3] Installing cmake and building llama.cpp (needed for conversion + quantize binaries) ==="
if ! command -v cmake &>/dev/null; then
    sudo apt-get install -y cmake
fi
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git pull --ff-only

# Build with cmake (uses all available cores)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

echo "=== [3/3] Installing llama.cpp Python conversion requirements ==="
pip install -r "$LLAMA_DIR/requirements.txt"

echo ""
echo "Setup complete."
echo "  llama.cpp built at: $LLAMA_DIR"
echo "  Binaries: $LLAMA_DIR/build/bin/"
echo ""
echo "Next: python scripts/llm_eval/download_convert.py"
