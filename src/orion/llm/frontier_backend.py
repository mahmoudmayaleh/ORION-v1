"""Hardened frontier (API) backend for §Q ORION-frontier — Claude Sonnet via the
Anthropic OpenAI-compatible endpoint.

Adds the four API-path pins the local llama.cpp path does not need
(SCOPING_Q_API_INTEGRATION_2026-07-15.md; §Q pilot ruling 2026-07-15):

  1. Key safety   — the API key is read from ORION_FRONTIER_API_KEY, never
                    written; `assert_key_absent` guards anything serialized.
  2. Model pin    — the dated snapshot in config.model is authoritative; the
                    per-call response.model is captured and MUST match the pin,
                    else it is an api-fail-class HALT, never a silent swap.
  3. Cost cap     — a running $ tally (tokens × rate); refuse-to-fire above the
                    cap (raise CostCapExceeded), same discipline as the :8000
                    and dirty-tree guards.
  4. Retry policy — transient API failures get bounded exponential backoff; on
                    exhaustion the call raises ApiFailError so the caller can
                    tag the arrival `api-fail` (distinct from `structural`), and
                    availability noise never masquerades as plan quality.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from orion.llm.llm_backend import LLMConfig, LLMBackend

logger = logging.getLogger(__name__)

ANTHROPIC_OPENAI_BASE_URL = "https://api.anthropic.com/v1/"
FRONTIER_KEY_ENV = "ORION_FRONTIER_API_KEY"

# Sonnet list price (per 1M tokens); recorded in config so per-cell $ is derived
# from measured tokens. Confirm current rates at pilot time.
SONNET_USD_PER_1M_INPUT = 3.0
SONNET_USD_PER_1M_OUTPUT = 15.0

# Provisional pilot-only cap (tens of dollars — far above the ~$0.30 estimate,
# well below the ~$100 cache-never-warms runaway). The GRID cap is set at the
# pilot-verdict checkpoint from measured tokens × 15 cells × 1.5.
PILOT_COST_CAP_USD = 10.0


class CostCapExceeded(RuntimeError):
    """Raised before a call that would cross the dollar cap. Refuse-to-fire."""


class ApiFailError(RuntimeError):
    """Raised after bounded retries are exhausted, or on a model-pin mismatch.

    The caller tags the arrival `api-fail` (a distinct reject reason, NOT
    plan-quality and NOT part of the malformed-output void trigger).
    """


@dataclass
class CostMeter:
    """Running token/dollar tally with a hard cap."""

    cap_usd: float = PILOT_COST_CAP_USD
    in_per_1m: float = SONNET_USD_PER_1M_INPUT
    out_per_1m: float = SONNET_USD_PER_1M_OUTPUT
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def spent_usd(self) -> float:
        return (self.input_tokens / 1e6) * self.in_per_1m + \
               (self.output_tokens / 1e6) * self.out_per_1m

    def would_exceed(self) -> bool:
        return self.spent_usd >= self.cap_usd

    def add(self, in_tok: int, out_tok: int) -> None:
        self.input_tokens += int(in_tok or 0)
        self.output_tokens += int(out_tok or 0)
        self.calls += 1

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "spent_usd": round(self.spent_usd, 4),
            "cap_usd": self.cap_usd,
        }


def make_frontier_config(model_snapshot: str, temperature: float = 0.0,
                         max_tokens: int = 2048, n_ctx: int = 200000) -> LLMConfig:
    """Build the Sonnet LLMConfig from the environment. Fail loud if the key is
    unset — a missing key is a pre-run error, not a mid-run api-fail."""
    key = os.environ.get(FRONTIER_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{FRONTIER_KEY_ENV} is not set. Source ~/ORION/.env.frontier before "
            f"launching the frontier approach (see SCOPING_Q_API_INTEGRATION §7)."
        )
    return LLMConfig(
        base_url=ANTHROPIC_OPENAI_BASE_URL,
        api_key=key,
        model=model_snapshot,          # dated snapshot, pinned at pilot (ruling 4)
        temperature=temperature,        # 0.0 for determinism
        max_tokens=max_tokens,
        n_ctx=n_ctx,
    )


def assert_key_absent(payload, key_value: str, where: str = "output") -> None:
    """Guard: the API key must never appear in anything that gets written
    (result JSON, §O.9 telemetry sidecars, logs). Call before every serialize."""
    if not key_value:
        return
    import json as _json
    try:
        blob = payload if isinstance(payload, str) else _json.dumps(payload, default=str)
    except (TypeError, ValueError):
        blob = str(payload)
    if key_value in blob:
        raise RuntimeError(f"API key leaked into {where} — refusing to write.")


class FrontierBackend(LLMBackend):
    """LLMBackend for the Anthropic OpenAI-compat endpoint with the four pins.

    Drop-in for AgentB (same `complete` signature + `last_finish_reason` /
    `last_prompt_tokens` attributes), so the existing Agent B plan path is
    unchanged; only the transport is hardened.
    """

    def __init__(self, config: LLMConfig, cost_meter: CostMeter,
                 max_retries: int = 4, backoff_base_s: float = 1.5) -> None:
        super().__init__(config)
        self.cost = cost_meter
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.last_model_id: str | None = None
        self.schema_retry_count: int = 0   # incremented by the caller on parse/validate fail
        self._pin = config.model
        # Endpoint-capability latches. Some models (e.g. claude-sonnet-5) reject
        # `temperature`; the OpenAI-compat endpoint may reject `response_format`.
        # We discover this on the FIRST call (one 400, unbilled), latch the flag
        # off, and never send that param again — no per-call wasted round-trip.
        self._send_temperature = True
        self._send_response_format = True

    def complete(self, system_prompt: str, user_message: str,
                 temperature: float | None = None, max_tokens: int | None = None,
                 response_format: dict | None = None) -> str:
        # Pin 3: refuse-to-fire above the cap, checked BEFORE the call.
        if self.cost.would_exceed():
            raise CostCapExceeded(
                f"cost cap ${self.cost.cap_usd} reached (spent ${self.cost.spent_usd:.2f}); "
                f"refusing further frontier calls."
            )

        base_kwargs: dict = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        # Determinism via temperature=0 WHERE the model supports it; some models
        # (Sonnet 5) reject the param entirely — latched off after discovery.
        if self._send_temperature:
            base_kwargs["temperature"] = self.config.temperature if temperature is None else temperature
        # Strict schema-constrained output (grammar-decoding equivalent) where
        # supported; else the Pydantic hard validator downstream is the guard.
        if response_format is not None and self._send_response_format:
            base_kwargs["response_format"] = response_format

        transient = 0
        while True:
            try:
                resp = self._client.chat.completions.create(**dict(base_kwargs))
                break
            except Exception as e:  # noqa: BLE001 — transport/availability, not plan quality
                msg = str(e).lower()
                # Request-construction fallbacks: latch the param off and retry
                # immediately (no backoff, unbilled 400). Only once per param.
                if "temperature" in msg and self._send_temperature:
                    self._send_temperature = False
                    base_kwargs.pop("temperature", None)
                    logger.warning("frontier: temperature unsupported for %s — dropping it",
                                   self.config.model)
                    continue
                if "response_format" in msg and self._send_response_format:
                    self._send_response_format = False
                    base_kwargs.pop("response_format", None)
                    logger.warning("frontier: response_format unsupported — dropping it "
                                   "(Pydantic hard-validates downstream)")
                    continue
                # Genuinely transient (429 / 5xx / timeout) → bounded backoff.
                if _is_transient(e) and transient < self.max_retries:
                    time.sleep(self.backoff_base_s * (2 ** transient))
                    transient += 1
                    logger.warning("frontier_api_transient attempt=%d err=%s", transient, str(e)[:160])
                    continue
                # Any other 4xx (bad request we can't fix) → fail fast, api-fail.
                raise ApiFailError(f"frontier API failed: {e}") from e

        choice = resp.choices[0]
        # Pin 2: capture + verify the model snapshot; a swap is api-fail, not silent.
        self.last_model_id = getattr(resp, "model", None)
        if self.last_model_id and self._pin and not _snapshot_matches(self.last_model_id, self._pin):
            raise ApiFailError(
                f"model pin mismatch: pinned '{self._pin}' but API served "
                f"'{self.last_model_id}' — halting rather than swapping models."
            )

        # Telemetry + cost (Pin 3 tally) from usage.
        self.last_finish_reason = choice.finish_reason
        usage = getattr(resp, "usage", None)
        self.last_prompt_tokens = getattr(usage, "prompt_tokens", None)
        self.last_completion_tokens = getattr(usage, "completion_tokens", None)
        self.finish_reason_counts[choice.finish_reason] = \
            self.finish_reason_counts.get(choice.finish_reason, 0) + 1
        if self.last_prompt_tokens is not None:
            self.max_prompt_tokens_seen = max(self.max_prompt_tokens_seen, self.last_prompt_tokens)
        self.cost.add(self.last_prompt_tokens or 0, self.last_completion_tokens or 0)

        return (choice.message.content or "").strip()


def _is_transient(e: Exception) -> bool:
    """True only for retryable availability errors: 429 / 5xx / timeout / connection.
    A plain 400 invalid_request is NOT transient — it fails fast (api-fail)."""
    s = str(e).lower()
    if "429" in s or "rate limit" in s or "overloaded" in s:
        return True
    if "timeout" in s or "timed out" in s or "connection" in s:
        return True
    return any(code in s for code in ("500", "502", "503", "504", "529"))


def _snapshot_matches(served: str, pinned: str) -> bool:
    """Served model must be the pinned snapshot (allow the provider echoing a
    longer fully-qualified id that startswith the pin, or vice versa)."""
    served, pinned = served.strip(), pinned.strip()
    return served == pinned or served.startswith(pinned) or pinned.startswith(served)
