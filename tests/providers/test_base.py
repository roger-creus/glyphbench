import pytest

from glyphbench.providers.base import (
    LLMClient,
    LLMResponse,
    ProviderError,
    ProviderInvalidRequest,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)


def test_llm_response_re_exported():
    r = LLMResponse(
        text="hi",
        tokens_in=1,
        tokens_out=1,
        tokens_reasoning=0,
        dollar_cost=0.0,
        latency_s=0.0,
        provider="mock",
        provider_request_id=None,
        raw_response={},
    )
    assert r.text == "hi"


def test_exception_hierarchy_is_provider_error():
    assert issubclass(ProviderRateLimit, ProviderError)
    assert issubclass(ProviderTransient, ProviderError)
    assert issubclass(ProviderInvalidRequest, ProviderError)
    assert issubclass(ProviderTimeout, ProviderError)


def test_llm_client_is_protocol():
    # Structural protocol: any class with matching members is an LLMClient
    class Fake:
        model_id = "x"
        provider = "fake"

        async def complete(self, system_prompt, user_prompt, *, temperature,
                           max_output_tokens, response_format, seed):  # noqa: ARG002
            return LLMResponse(
                text="", tokens_in=0, tokens_out=0, tokens_reasoning=0,
                dollar_cost=0.0, latency_s=0.0, provider="fake",
                provider_request_id=None, raw_response={},
            )

    f = Fake()
    assert isinstance(f, LLMClient)


def test_raising_provider_errors_preserves_messages():
    with pytest.raises(ProviderRateLimit, match="rate limited"):
        raise ProviderRateLimit("rate limited: 429")
