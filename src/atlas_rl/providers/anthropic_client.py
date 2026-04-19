"""Anthropic client wrapping anthropic.AsyncAnthropic via dependency injection."""

from __future__ import annotations

import time
from typing import Any

import anthropic

from atlas_rl.providers.base import (
    LLMResponse,
    ProviderInvalidRequest,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)
from atlas_rl.providers.pricing import Pricing


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        *,
        sdk_client: Any,  # anthropic.AsyncAnthropic
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
        # Anthropic's Messages API: system is a top-level field, not a message role.
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        # Anthropic has no native JSON-schema response_format; we rely on system
        # prompt to instruct JSON output and the harness parser to extract it.

        start = time.perf_counter()
        try:
            response = await self._sdk.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise ProviderRateLimit(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise ProviderTransient(str(e)) from e
        except anthropic.APITimeoutError as e:
            raise ProviderTimeout(str(e)) from e
        except anthropic.BadRequestError as e:
            raise ProviderInvalidRequest(str(e)) from e
        except anthropic.APIStatusError as e:
            if 500 <= e.status_code < 600:
                raise ProviderTransient(str(e)) from e
            raise ProviderInvalidRequest(str(e)) from e
        latency_s = time.perf_counter() - start

        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts)
        usage = response.usage
        tokens_in = int(getattr(usage, "input_tokens", 0))
        tokens_out = int(getattr(usage, "output_tokens", 0))
        tokens_reasoning = 0
        # If the model uses extended thinking, `usage` may include
        # `output_tokens_details` with `reasoning_tokens`. Support that when present.
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            tokens_reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)

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
            provider_request_id=getattr(response, "id", None),
            raw_response=raw,
        )
