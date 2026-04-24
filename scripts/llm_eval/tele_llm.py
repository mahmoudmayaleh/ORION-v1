"""
Inference wrapper for the quantized LLama-3-8B-Tele-it model.

Can be run directly for a quick smoke test:
    python scripts/llm_eval/tele_llm.py

Or imported in agent code:
    from scripts.llm_eval.tele_llm import TeleLLM
    llm = TeleLLM()
    response = llm.complete("What is network slicing?")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_PATH = Path.home() / "llm_models" / "tele-llm" / "tele-llm-q4km.gguf"

# LLaMA-3 chat template tokens
# Note: llama-cpp-python adds <|begin_of_text|> (BOS) automatically; omit it here.
_SYS_START = "<|start_header_id|>system<|end_header_id|>\n\n"
_SYS_END   = "<|eot_id|>"
_USR_START = "<|start_header_id|>user<|end_header_id|>\n\n"
_USR_END   = "<|eot_id|>"
_AST_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"


@dataclass
class TeleLLMConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    n_ctx: int = 8192       # model's full context window
    n_threads: int = 4      # physical cores on i7-1165G7
    n_batch: int = 512
    temperature: float = 0.1
    max_tokens: int = 512
    system_prompt: str = (
        "You are a telecom network expert specialized in 6G network slicing "
        "and resource orchestration. Answer concisely and precisely."
    )


class TeleLLM:
    """Thin wrapper around llama-cpp-python for the quantized Tele LLM."""

    def __init__(self, config: TeleLLMConfig | None = None) -> None:
        self.config = config or TeleLLMConfig()
        self._llm = self._load()

    def _load(self):
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            ) from e

        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.config.model_path}\n"
                "Run: python scripts/llm_eval/download_convert.py"
            )

        print(f"Loading model from {self.config.model_path} ...")
        return Llama(
            model_path=str(self.config.model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_batch=self.config.n_batch,
            verbose=False,
        )

    def _build_prompt(self, user_message: str) -> str:
        return (
            f"{_SYS_START}{self.config.system_prompt}{_SYS_END}"
            f"{_USR_START}{user_message}{_USR_END}"
            f"{_AST_START}"
        )

    def complete(self, user_message: str, **kwargs) -> str:
        """Return the model's text response to a user message."""
        prompt = self._build_prompt(user_message)
        result = self._llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    test_prompts = [
        "What is the difference between eMBB, URLLC, and mMTC network slices?",
        "A 6G network slice requires 500 Mbps bandwidth and 1 ms latency. "
        "Which physical resource blocks should be prioritised?",
    ]

    llm = TeleLLM()

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"USER: {prompt}")
        t0 = time.perf_counter()
        response = llm.complete(prompt)
        elapsed = time.perf_counter() - t0
        print(f"ASSISTANT ({elapsed:.1f}s): {response}")
