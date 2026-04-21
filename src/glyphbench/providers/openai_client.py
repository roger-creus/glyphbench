"""OpenAI client wrapping openai.AsyncOpenAI via dependency injection."""

from __future__ import annotations

import time
from typing import Any

import openai

from glyphbench.providers.base import (
    LLMResponse,
    ProviderInvalidRequest,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)
from glyphbench.providers.pricing import Pricing


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        *,
        sdk_client: Any,  # openai.AsyncOpenAI, but we accept any duck-type for tests
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
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if response_format is not None:
            # Request JSON-object mode; schema enforcement is best-effort
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            response = await self._sdk.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise ProviderRateLimit(str(e)) from e
        except openai.APIConnectionError as e:
            raise ProviderTransient(str(e)) from e
        except openai.APITimeoutError as e:
            raise ProviderTimeout(str(e)) from e
        except openai.BadRequestError as e:
            raise ProviderInvalidRequest(str(e)) from e
        except openai.APIStatusError as e:
            if 500 <= e.status_code < 600:
                raise ProviderTransient(str(e)) from e
            raise ProviderInvalidRequest(str(e)) from e
        latency_s = time.perf_counter() - start

        text = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0))
        tokens_out = int(getattr(usage, "completion_tokens", 0))
        tokens_reasoning = 0
        details = getattr(usage, "completion_tokens_details", None)
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
