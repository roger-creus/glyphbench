import httpx
import pytest

from atlas_rl.providers.base import LLMResponse, ProviderInvalidRequest, ProviderRateLimit
from atlas_rl.providers.pricing import ModelRates, Pricing
from atlas_rl.providers.vllm_client import VllmClient


def _pricing_stub() -> Pricing:
    return Pricing(
        version="test",
        providers={
            "vllm": {
                "*": ModelRates(
                    input_per_1m_tokens=0.0,
                    output_per_1m_tokens=0.0,
                    reasoning_per_1m_tokens=0.0,
                )
            }
        },
    )


def _fake_success_response(request: httpx.Request) -> httpx.Response:
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"action": "NORTH"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 45,
        },
        "id": "vllm-test-1",
    }
    return httpx.Response(200, json=body)


@pytest.mark.asyncio
async def test_vllm_client_happy_path():
    transport = httpx.MockTransport(_fake_success_response)
    http = httpx.AsyncClient(transport=transport, base_url="http://vllm.test/v1")
    client = VllmClient(
        model_id="test-model",
        http_client=http,
        pricing=_pricing_stub(),
    )
    r: LLMResponse = await client.complete(
        system_prompt="sys",
        user_prompt="user",
        temperature=0.0,
        max_output_tokens=128,
        response_format=None,
        seed=None,
    )
    assert r.text == '{"action": "NORTH"}'
    assert r.tokens_in == 123
    assert r.tokens_out == 45
    assert r.dollar_cost == 0.0
    assert r.provider == "vllm"
    assert r.provider_request_id == "vllm-test-1"
    assert r.latency_s >= 0.0


@pytest.mark.asyncio
async def test_vllm_client_rate_limit_raises_provider_rate_limit():
    def fake(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    transport = httpx.MockTransport(fake)
    http = httpx.AsyncClient(transport=transport, base_url="http://vllm.test/v1")
    client = VllmClient(model_id="test-model", http_client=http, pricing=_pricing_stub())
    with pytest.raises(ProviderRateLimit):
        await client.complete(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
            max_output_tokens=128,
            response_format=None,
            seed=None,
        )


@pytest.mark.asyncio
async def test_vllm_client_4xx_raises_invalid_request():
    def fake(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad input"})

    transport = httpx.MockTransport(fake)
    http = httpx.AsyncClient(transport=transport, base_url="http://vllm.test/v1")
    client = VllmClient(model_id="test-model", http_client=http, pricing=_pricing_stub())
    with pytest.raises(ProviderInvalidRequest):
        await client.complete(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.0,
            max_output_tokens=128,
            response_format=None,
            seed=None,
        )
