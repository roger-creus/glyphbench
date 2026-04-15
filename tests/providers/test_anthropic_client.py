from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rl_world_ascii.providers.anthropic_client import AnthropicClient
from rl_world_ascii.providers.base import ProviderInvalidRequest, ProviderRateLimit
from rl_world_ascii.providers.pricing import ModelRates, Pricing


def _pricing() -> Pricing:
    return Pricing(
        version="test",
        providers={
            "anthropic": {
                "claude-test": ModelRates(
                    input_per_1m_tokens=3.0,
                    output_per_1m_tokens=15.0,
                    reasoning_per_1m_tokens=15.0,
                )
            }
        },
    )


def _fake_message(text: str, input_tokens: int, output_tokens: int, request_id: str):
    return SimpleNamespace(
        id=request_id,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        model_dump=lambda: {"id": request_id},
    )


def _raise_via_new(exc_cls, message: str):
    """Raise an SDK exception without calling __init__.

    Newer anthropic/openai SDK versions require a real httpx.Response with a
    `.request` attribute in their exception constructors. We bypass that
    entirely: create the instance via __new__, set the message with
    Exception.__init__, and raise.
    """
    err = exc_cls.__new__(exc_cls)
    Exception.__init__(err, message)
    raise err


@pytest.mark.asyncio
async def test_anthropic_client_happy_path():
    sdk = SimpleNamespace(messages=SimpleNamespace())
    sdk.messages.create = AsyncMock(
        return_value=_fake_message('{"action": "NORTH"}', 1_000_000, 500_000, "msg-1")
    )
    client = AnthropicClient(sdk_client=sdk, model_id="claude-test", pricing=_pricing())
    r = await client.complete(
        system_prompt="sys",
        user_prompt="user",
        temperature=0.0,
        max_output_tokens=256,
        response_format=None,
        seed=None,
    )
    assert r.text == '{"action": "NORTH"}'
    assert r.tokens_in == 1_000_000
    assert r.tokens_out == 500_000
    assert r.dollar_cost == pytest.approx(3.0 + 7.5)  # 1M * $3 + 0.5M * $15
    assert r.provider == "anthropic"
    assert r.provider_request_id == "msg-1"


@pytest.mark.asyncio
async def test_anthropic_client_rate_limit_mapped():
    import anthropic

    sdk = SimpleNamespace(messages=SimpleNamespace())

    async def raise_rate(**kwargs):
        _raise_via_new(anthropic.RateLimitError, "rate limited")

    sdk.messages.create = raise_rate
    client = AnthropicClient(sdk_client=sdk, model_id="claude-test", pricing=_pricing())
    with pytest.raises(ProviderRateLimit):
        await client.complete(
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_output_tokens=1, response_format=None, seed=None,
        )


@pytest.mark.asyncio
async def test_anthropic_client_bad_request_mapped():
    import anthropic

    sdk = SimpleNamespace(messages=SimpleNamespace())

    async def raise_bad(**kwargs):
        _raise_via_new(anthropic.BadRequestError, "bad")

    sdk.messages.create = raise_bad
    client = AnthropicClient(sdk_client=sdk, model_id="claude-test", pricing=_pricing())
    with pytest.raises(ProviderInvalidRequest):
        await client.complete(
            system_prompt="s", user_prompt="u",
            temperature=0.0, max_output_tokens=1, response_format=None, seed=None,
        )
