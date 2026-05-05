"""Unified LLM backend abstraction for Agent A and Agent B.

Supports any OpenAI-compatible API (vLLM, Anthropic, OpenAI, llama-cpp-server)
via a single interface. The backend is selected by config — no code changes
needed to swap providers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for an OpenAI-compatible LLM endpoint."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "default"
    temperature: float = 0.05
    max_tokens: int = 2048
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    embedding_backend: Literal["openai_api", "local_st"] = "local_st"


class LLMBackend:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = self._make_client()

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
    ) -> str:
        """Send a chat completion request and return the assistant's text."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()


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
