from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from glyphbench.providers.gemini_client import GeminiClient
from glyphbench.providers.pricing import ModelRates, Pricing


def _pricing() -> Pricing:
    return Pricing(
        version="test",
        providers={
            "google": {
                "gemini-test": ModelRates(
                    input_per_1m_tokens=1.25,
                    output_per_1m_tokens=5.0,
                    reasoning_per_1m_tokens=None,
                )
            }
        },
    )


def _fake_gemini_response(text: str, prompt_tokens: int, output_tokens: int, request_id: str):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            total_token_count=prompt_tokens + output_tokens,
        ),
        response_id=request_id,
        model_dump=lambda: {"id": request_id},
    )


@pytest.mark.asyncio
async def test_gemini_client_happy_path():
    aio_models = SimpleNamespace()
    aio_models.generate_content = AsyncMock(
        return_value=_fake_gemini_response('{"action": "NORTH"}', 2_000_000, 500_000, "resp-1")
    )
    sdk = SimpleNamespace(aio=SimpleNamespace(models=aio_models))
    client = GeminiClient(sdk_client=sdk, model_id="gemini-test", pricing=_pricing())
    r = await client.complete(
        system_prompt="sys",
        user_prompt="user",
        temperature=0.5,
        max_output_tokens=512,
        response_format=None,
        seed=None,
    )
    assert r.text == '{"action": "NORTH"}'
    assert r.tokens_in == 2_000_000
    assert r.tokens_out == 500_000
    # 2M * $1.25 + 0.5M * $5 = $2.5 + $2.5 = $5
    assert r.dollar_cost == pytest.approx(5.0)
    assert r.provider == "google"
