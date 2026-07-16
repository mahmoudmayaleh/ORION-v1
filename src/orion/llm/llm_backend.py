"""Unified LLM backend abstraction for Agent A and Agent B.

Supports any OpenAI-compatible API (vLLM, Anthropic, OpenAI, llama-cpp-server)
via a single interface. The backend is selected by config — no code changes
needed to swap providers.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


class EmptyCompletionError(RuntimeError):
    """A completion returned zero tokens / empty content. Raised loudly so an
    empty response can never flow downstream as if it were model output (the
    2026-07-15 wedge: the server answered HTTP 200 with 0 tokens and E.1 happily
    consumed empty plans into a plausible-looking 18/100). Crash > silent corruption."""


class LocalLLMBusyError(RuntimeError):
    """Another local-LLM job already holds the single-slot server lock. Refuse to
    start rather than wedge the server with concurrent load (the incident's cause)."""


# ── Single-slot local-LLM lock (durable form of "never run two local-LLM jobs
# concurrently"). One process holds it; a second refuses loudly. Localhost only —
# frontier/API backends are never locked. Re-entrant within a PID (a runner may
# build several LLMBackends). Stale locks (dead PID) are stolen. Opt out with
# ORION_LLM_LOCK_DISABLE=1 for deliberate multi-port parallelism. Same discipline
# class as the :8000 oracle guard and the dirty-tree guard. ──────────────────────
_LOCK_PATH = os.path.join(tempfile.gettempdir(), "orion_local_llm.lock")
_LOCK_OWNED_BY_US = False  # this PID holds it (module-global => re-entrant per process)


def _is_local(base_url: str) -> bool:
    return ("localhost" in base_url) or ("127.0.0.1" in base_url) or ("0.0.0.0" in base_url)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
    except PermissionError:
        return True  # exists, owned by someone else


def acquire_local_llm_lock() -> None:
    """Acquire the single-slot local-LLM lock or raise LocalLLMBusyError."""
    global _LOCK_OWNED_BY_US
    if os.environ.get("ORION_LLM_LOCK_DISABLE"):
        return
    if _LOCK_OWNED_BY_US:
        return  # re-entrant within this process
    while True:
        try:
            fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            _LOCK_OWNED_BY_US = True
            atexit.register(release_local_llm_lock)
            logger.info("local-LLM lock acquired (pid=%d, %s)", os.getpid(), _LOCK_PATH)
            return
        except FileExistsError:
            try:
                other = int(open(_LOCK_PATH).read().strip() or "-1")
            except (ValueError, OSError):
                other = -1
            if other == os.getpid():
                _LOCK_OWNED_BY_US = True
                return
            if other < 0 or not _pid_alive(other):
                logger.warning("stealing stale local-LLM lock (dead pid=%s)", other)
                try:
                    os.unlink(_LOCK_PATH)
                except OSError:
                    pass
                continue
            raise LocalLLMBusyError(
                f"\n*** REFUSING TO START: another local-LLM job holds the lock "
                f"(pid={other}, {_LOCK_PATH}).\n"
                f"*** The local llama.cpp server is single-slot; concurrent jobs wedge it.\n"
                f"*** Wait for pid={other} to finish, or `kill {other}` if it is dead, "
                f"or set ORION_LLM_LOCK_DISABLE=1 to override (only if you know the server "
                f"can take it).")


def release_local_llm_lock() -> None:
    global _LOCK_OWNED_BY_US
    if not _LOCK_OWNED_BY_US:
        return
    try:
        if os.path.exists(_LOCK_PATH) and open(_LOCK_PATH).read().strip() == str(os.getpid()):
            os.unlink(_LOCK_PATH)
    except OSError:
        pass
    _LOCK_OWNED_BY_US = False


@dataclass
class LLMConfig:
    """Configuration for an OpenAI-compatible LLM endpoint."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "default"
    temperature: float = 0.05
    max_tokens: int = 2048
    # MUST match the served llama.cpp `--n_ctx`. Used for the headroom guardrail:
    # memory-augmented Agent B prompts can approach the window and silently
    # truncate the completion (context-exhaustion looks like a JSON parse failure
    # downstream). Pinned at 8192 = LLaMA-3-8B native; see Amendment 3.
    n_ctx: int = 8192
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    embedding_backend: Literal["openai_api", "local_st"] = "local_st"


class LLMBackend:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Single-slot guard: a local-LLM backend acquires the process lock so a
        # second concurrent local-LLM job refuses instead of wedging the server.
        # Frontier/API backends (non-localhost) are never locked.
        if _is_local(config.base_url):
            acquire_local_llm_lock()
        self._client = self._make_client()
        # Per-call telemetry from the most recent completion (guardrails read these).
        self.last_finish_reason: str | None = None
        self.last_prompt_tokens: int | None = None
        self.last_completion_tokens: int | None = None
        # Cumulative telemetry across the backend's lifetime (runner reports these
        # and gates the warm-up worst case on max_prompt_tokens_seen).
        self.finish_reason_counts: dict[str, int] = {}
        self.max_prompt_tokens_seen: int = 0

    def _make_client(self):
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from e

        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

    def embed(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Uses the configured embedding_backend:
          - "openai_api": calls /v1/embeddings via the OpenAI client.
          - "local_st": loads via sentence_transformers.SentenceTransformer.

        Args:
            texts: Texts to embed.
            model: Override embedding model name. Uses config default if None.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If model weights are unavailable.
        """
        model_name = model or self.config.embedding_model

        if self.config.embedding_backend == "openai_api":
            return self._embed_openai(texts, model_name)
        else:
            return self._embed_local_st(texts, model_name)

    def _embed_openai(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed via OpenAI-compatible /v1/embeddings endpoint."""
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    def _embed_local_st(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed via local sentence-transformers model."""
        if not hasattr(self, "_st_models"):
            self._st_models: dict = {}

        if model not in self._st_models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence_transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from e

            try:
                self._st_models[model] = SentenceTransformer(model)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{model}'. "
                    f"Ensure model weights are downloaded. Error: {e}"
                ) from e

        st_model = self._st_models[model]
        embeddings = st_model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant's text.

        If response_format is given (e.g. {"type": "json_object"}), it is
        forwarded to the endpoint. Against a local llama.cpp server this
        enforces a JSON grammar at decode time, so the model cannot emit
        malformed JSON — the format-drift failure mode on long VNF chains.
        """
        kwargs: dict = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        eff_max_tokens = max_tokens or self.config.max_tokens
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # ── Guardrails (Amendment 3) ──────────────────────────────────────────
        # A memory-augmented prompt can approach n_ctx and truncate the
        # completion. With grammar-constrained decoding that truncation yields
        # cut-off (invalid) JSON, which downstream looked like a plain parse
        # failure — and it was arm-asymmetric (only the K^B+M^B arms hit it).
        # Record finish_reason + usage so truncation is a *distinct, counted*
        # signal, and check headroom against the configured window.
        self.last_finish_reason = choice.finish_reason
        usage = getattr(response, "usage", None)
        self.last_prompt_tokens = getattr(usage, "prompt_tokens", None)
        self.last_completion_tokens = getattr(usage, "completion_tokens", None)
        self.finish_reason_counts[choice.finish_reason] = (
            self.finish_reason_counts.get(choice.finish_reason, 0) + 1
        )
        if self.last_prompt_tokens is not None:
            self.max_prompt_tokens_seen = max(
                self.max_prompt_tokens_seen, self.last_prompt_tokens
            )
        if choice.finish_reason == "length":
            logger.warning(
                "llm_completion_truncated",
                extra={
                    "prompt_tokens": self.last_prompt_tokens,
                    "completion_tokens": self.last_completion_tokens,
                    "n_ctx": self.config.n_ctx,
                },
            )
        # Early-warning: fewer than _COMPLETION_RESERVE tokens of window remain
        # for the completion. A realistic plan is a few hundred tokens, so use a
        # reserve rather than the max_tokens ceiling (which would over-warn).
        _COMPLETION_RESERVE = 1024
        if (
            self.last_prompt_tokens is not None
            and self.last_prompt_tokens > self.config.n_ctx - _COMPLETION_RESERVE
        ):
            logger.warning(
                "llm_context_headroom_low",
                extra={
                    "prompt_tokens": self.last_prompt_tokens,
                    "n_ctx": self.config.n_ctx,
                    "headroom": self.config.n_ctx - self.last_prompt_tokens,
                },
            )
        content = (choice.message.content or "").strip()
        # Empty-completion assert: HTTP 200 with 0 tokens / empty body is exactly
        # the wedge signature a process-liveness check cannot see. Raise loudly so
        # an empty response can never be consumed downstream as model output.
        if not content or self.last_completion_tokens == 0:
            raise EmptyCompletionError(
                f"LLM returned an empty completion (completion_tokens="
                f"{self.last_completion_tokens}, finish_reason={self.last_finish_reason}, "
                f"base_url={self.config.base_url}). The server is likely WEDGED — restart it "
                f"(scripts/start_llm_gpu.sh) and re-run. Refusing to emit empty output.")
        return content


def extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM output text.

    Handles markdown code fences and surrounding prose.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find outermost braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model response:\n{text[:500]}")
