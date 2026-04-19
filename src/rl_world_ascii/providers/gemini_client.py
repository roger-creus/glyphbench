"""Gemini client wrapping google.genai.Client via dependency injection.

Uses the async-in-SDK pattern: sdk.aio.models.generate_content(...).
"""

from __future__ import annotations

import time
from typing import Any

from rl_world_ascii.providers.base import (
    LLMResponse,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)
from rl_world_ascii.providers.pricing import Pricing


class GeminiClient:
    provider = "google"

    def __init__(
        self,
        *,
        sdk_client: Any,  # google.genai.Client
        model_id: str,
        pricing: Pricing,
    ) -> None:
        self._sdk = sdk_client
        self.model_id = model_id
        self._pricing = pricing

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None,
        seed: int | None,
    ) -> LLMResponse:
        # Gemini takes a single `contents` field; system prompt goes in config.
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "system_instruction": system_prompt,
        }
        if response_format is not None:
            config["response_mime_type"] = "application/json"

        start = time.perf_counter()
        try:
            response = await self._sdk.aio.models.generate_content(
                model=self.model_id,
                contents=user_prompt,
                config=config,
            )
        except Exception as e:  # noqa: BLE001  # google-genai raises SDK-specific errors; map loosely
            msg = str(e)
            lower = msg.lower()
            if "rate" in lower or "429" in lower or "quota" in lower:
                raise ProviderRateLimit(msg) from e
            if "timeout" in lower:
                raise ProviderTimeout(msg) from e
            if "500" in lower or "503" in lower or "unavailable" in lower:
                raise ProviderTransient(msg) from e
            raise ProviderTransient(msg) from e
        latency_s = time.perf_counter() - start

        text = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        tokens_in = int(getattr(usage, "prompt_token_count", 0)) if usage else 0
        tokens_out = int(getattr(usage, "candidates_token_count", 0)) if usage else 0
        tokens_reasoning = 0  # Gemini reasoning tokens surfaced via `thoughts_token_count` on some models
        if usage is not None:
            tokens_reasoning = int(getattr(usage, "thoughts_token_count", 0) or 0)

        cost = self._pricing.compute_cost(
            provider=self.provider,
            model_id=self.model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_reasoning=tokens_reasoning,
        )
        try:
            raw = response.model_dump()
        except AttributeError:
            raw = {}
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_reasoning=tokens_reasoning,
            dollar_cost=cost,
            latency_s=latency_s,
            provider=self.provider,
            provider_request_id=getattr(response, "response_id", None),
            raw_response=raw,
        )
