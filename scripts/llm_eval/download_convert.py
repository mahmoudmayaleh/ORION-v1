"""
Download LLama-3-8B-Tele-it from HuggingFace, convert to GGUF, and quantize to Q4_K_M.

Usage:
    python scripts/llm_eval/download_convert.py

Outputs:
    ~/llm_models/tele-llm/tele-llm-q4km.gguf  (~4.7 GB)
"""

import subprocess
import sys
from pathlib import Path

HF_REPO = "AliMaatouk/LLama-3-8B-Tele-it"
MODEL_DIR = Path.home() / "llm_models" / "tele-llm"
GGUF_F16 = MODEL_DIR / "tele-llm-f16.gguf"
GGUF_Q4KM = MODEL_DIR / "tele-llm-q4km.gguf"
LLAMA_CPP = Path.home() / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP / "build" / "bin" / "llama-quantize"


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def check_prereqs() -> None:
    if not CONVERT_SCRIPT.exists():
        sys.exit(
            f"llama.cpp not found at {LLAMA_CPP}. "
            "Run: bash scripts/llm_eval/setup.sh"
        )
    if not QUANTIZE_BIN.exists():
        sys.exit(
            f"quantize binary not found at {QUANTIZE_BIN}. "
            "Run: bash scripts/llm_eval/setup.sh"
        )


def download_model() -> None:
    print(f"\n=== Downloading {HF_REPO} to {MODEL_DIR} ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    snapshot_download(repo_id=HF_REPO, local_dir=str(MODEL_DIR))
    print("Download complete.")


def convert_to_gguf() -> None:
    if GGUF_F16.exists():
        print(f"\nF16 GGUF already exists: {GGUF_F16}, skipping conversion.")
        return
    print("\n=== Converting to GGUF (F16) ===")
    run([sys.executable, str(CONVERT_SCRIPT), str(MODEL_DIR),
         "--outfile", str(GGUF_F16), "--outtype", "f16"])


def quantize() -> None:
    if GGUF_Q4KM.exists():
        print(f"\nQuantized model already exists: {GGUF_Q4KM}, skipping.")
        return
    print("\n=== Quantizing to Q4_K_M (~4.7 GB) ===")
    run([str(QUANTIZE_BIN), str(GGUF_F16), str(GGUF_Q4KM), "Q4_K_M"])


def cleanup_f16() -> None:
    if GGUF_F16.exists():
        print(f"\nRemoving intermediate F16 file ({GGUF_F16.stat().st_size / 1e9:.1f} GB)...")
        GGUF_F16.unlink()
        print("Removed.")


if __name__ == "__main__":
    check_prereqs()
    download_model()
    convert_to_gguf()
    quantize()
    cleanup_f16()

    print(f"\nDone! Quantized model at:\n  {GGUF_Q4KM}")
    print("\nTest it with:\n  python scripts/llm_eval/tele_llm.py")
