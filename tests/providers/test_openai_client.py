from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from glyphbench.providers.base import LLMResponse, ProviderInvalidRequest, ProviderRateLimit
from glyphbench.providers.openai_client import OpenAIClient
from glyphbench.providers.pricing import ModelRates, Pricing


def _pricing() -> Pricing:
    return Pricing(
        version="test",
        providers={
            "openai": {
                "gpt-4o-test": ModelRates(
                    input_per_1m_tokens=2.0,
                    output_per_1m_tokens=10.0,
                    reasoning_per_1m_tokens=None,
                )
            }
        },
    )


def _fake_chat_completion(text: str, prompt_tokens: int, completion_tokens: int, request_id: str):
    return SimpleNamespace(
        id=request_id,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=None,
        ),
        model_dump=lambda: {"id": request_id, "text": text},
    )


@pytest.mark.asyncio
async def test_openai_client_happy_path():
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
    sdk.chat.completions.create = AsyncMock(
        return_value=_fake_chat_completion('{"action": "NORTH"}', 1_000_000, 500_000, "chatcmpl-1")
    )
    client = OpenAIClient(sdk_client=sdk, model_id="gpt-4o-test", pricing=_pricing())
    r: LLMResponse = await client.complete(
        system_prompt="sys",
        user_prompt="user",
        temperature=0.2,
        max_output_tokens=512,
        response_format=None,
        seed=42,
    )
    assert r.text == '{"action": "NORTH"}'
    assert r.tokens_in == 1_000_000
    assert r.tokens_out == 500_000
    assert r.dollar_cost == pytest.approx(7.0)  # 1M * $2 + 0.5M * $10
    assert r.provider == "openai"
    assert r.provider_request_id == "chatcmpl-1"


@pytest.mark.asyncio
async def test_openai_client_rate_limit_mapped():
    import openai

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))

    async def raise_rate_limit(**kwargs):
        # openai SDK 2.x requires a real httpx.Response (with a .request
        # attribute) for RateLimitError.__init__. Bypass __init__ entirely
        # to construct an instance that tests can raise without pulling in
        # httpx plumbing.
        err = openai.RateLimitError.__new__(openai.RateLimitError)
        Exception.__init__(err, "rate limited")
        raise err

    sdk.chat.completions.create = raise_rate_limit
    client = OpenAIClient(sdk_client=sdk, model_id="gpt-4o-test", pricing=_pricing())
    with pytest.raises(ProviderRateLimit):
        await client.complete(
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_output_tokens=1, response_format=None, seed=None,
        )


@pytest.mark.asyncio
async def test_openai_client_bad_request_mapped():
    import openai

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))

    async def raise_bad(**kwargs):
        # Same __init__ bypass as rate-limit test - see note above.
        err = openai.BadRequestError.__new__(openai.BadRequestError)
        Exception.__init__(err, "bad")
        raise err

    sdk.chat.completions.create = raise_bad
    client = OpenAIClient(sdk_client=sdk, model_id="gpt-4o-test", pricing=_pricing())
    with pytest.raises(ProviderInvalidRequest):
        await client.complete(
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_output_tokens=1, response_format=None, seed=None,
        )
