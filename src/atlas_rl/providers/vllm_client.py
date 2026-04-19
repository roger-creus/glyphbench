"""vLLM client using the OpenAI-compatible /v1/chat/completions endpoint.

We talk to vLLM directly via httpx instead of the openai SDK so that tests can
cleanly mock the HTTP layer with httpx.MockTransport.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from atlas_rl.providers.base import (
    LLMResponse,
    ProviderInvalidRequest,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)
from atlas_rl.providers.pricing import Pricing


class VllmClient:
    provider = "vllm"

    def __init__(
        self,
        *,
        model_id: str,
        http_client: httpx.AsyncClient,
        pricing: Pricing,
    ) -> None:
        self.model_id = model_id
        self._http = http_client
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
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        # vLLM supports OpenAI-style response_format with json_schema or json_object.
        # Keep it optional - many vLLM deployments don't enable schema-constrained output.
        if response_format is not None:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            response = await self._http.post("/chat/completions", json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(str(e)) from e
        except httpx.TransportError as e:
            raise ProviderTransient(str(e)) from e
        latency_s = time.perf_counter() - start

        if response.status_code == 429:
            raise ProviderRateLimit(f"vllm 429: {response.text}")
        if 500 <= response.status_code < 600:
            raise ProviderTransient(f"vllm {response.status_code}: {response.text}")
        if response.status_code >= 400:
            raise ProviderInvalidRequest(f"vllm {response.status_code}: {response.text}")

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderInvalidRequest(f"vllm response missing choices: {body}")
        text = choices[0]["message"]["content"] or ""
        usage = body.get("usage", {})
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        tokens_reasoning = int(usage.get("reasoning_tokens", 0))
        cost = self._pricing.compute_cost(
            provider=self.provider,
            model_id=self.model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_reasoning=tokens_reasoning,
        )
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_reasoning=tokens_reasoning,
            dollar_cost=cost,
            latency_s=latency_s,
            provider=self.provider,
            provider_request_id=str(body.get("id", "")) or None,
            raw_response=body,
        )
