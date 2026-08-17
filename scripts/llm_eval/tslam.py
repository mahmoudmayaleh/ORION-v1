"""
Inference wrapper for the TSLAM-4B model (HuggingFace Transformers).

Can be run directly for a quick smoke test:
    python scripts/llm_eval/tslam.py

Or imported in agent code:
    from scripts.llm_eval.tslam import TSLAMLLM
    llm = TSLAMLLM()
    response = llm.complete("What is network slicing?")
"""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "NetoAISolutions/TSLAM-4B"

# Phi-3 style chat template tokens used by TSLAM-4B
_SYS_START = "<|system|>\n"
_SYS_END   = "<|end|>\n"
_USR_START = "<|user|>\n"
_USR_END   = "<|end|>\n"
_AST_START = "<|assistant|>\n"


@dataclass
class TSLAMConfig:
    model_id: str = MODEL_ID
    max_new_tokens: int = 512
    temperature: float = 0.1
    system_prompt: str = (
        "You are a telecom network expert specialized in 6G network slicing "
        "and resource orchestration. Answer concisely and precisely."
    )


class TSLAMLLM:
    """Thin wrapper around HuggingFace Transformers for TSLAM-4B."""

    def __init__(self, config: TSLAMConfig | None = None) -> None:
        self.config = config or TSLAMConfig()
        self._tokenizer, self._model = self._load()

    def _load(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            import torch
        except ImportError as e:
            raise RuntimeError(
                "transformers or torch not installed. "
                "Run: pip install transformers torch accelerate"
            ) from e

        import json
        from pathlib import Path
        from huggingface_hub import hf_hub_download

        print(f"Loading TSLAM-4B from {self.config.model_id} ...")

        # TSLAM-4B was built with transformers 4.45.x which stored
        # original_max_position_embeddings at the top level of config.json.
        # Transformers 5.x requires it *inside* rope_scaling for longrope/su.
        # Patch the cached config.json in-place (once) so every loading path —
        # AutoConfig, AutoTokenizer — sees the corrected dict.
        cached_cfg = Path(hf_hub_download(self.config.model_id, "config.json"))
        config_dict = json.loads(cached_cfg.read_text())
        rs = config_dict.get("rope_scaling", {})
        if "original_max_position_embeddings" not in rs:
            rs["original_max_position_embeddings"] = config_dict.get(
                "original_max_position_embeddings", 4096
            )
            config_dict["rope_scaling"] = rs
            cached_cfg.write_text(json.dumps(config_dict, indent=4))
            print("Patched cached config.json for transformers 5.x compatibility.")

        cfg = AutoConfig.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            config=cfg,
            trust_remote_code=True,
            device_map="auto",
        )
        model.eval()
        return tokenizer, model

    def _build_prompt(self, user_message: str) -> str:
        return (
            f"{_SYS_START}{self.config.system_prompt}{_SYS_END}"
            f"{_USR_START}{user_message}{_USR_END}"
            f"{_AST_START}"
        )

    def complete(self, user_message: str, **kwargs) -> str:
        """Return the model's text response to a user message."""
        import torch

        prompt = self._build_prompt(user_message)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Smoke test — same prompts as tele_llm.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    test_prompts = [
        "What is the difference between eMBB, URLLC, and mMTC network slices?",
        "A 6G network slice requires 500 Mbps bandwidth and 1 ms latency. "
        "Which physical resource blocks should be prioritised?",
    ]

    llm = TSLAMLLM()

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"USER: {prompt}")
        t0 = time.perf_counter()
        response = llm.complete(prompt)
        elapsed = time.perf_counter() - t0
        print(f"ASSISTANT ({elapsed:.1f}s): {response}")
