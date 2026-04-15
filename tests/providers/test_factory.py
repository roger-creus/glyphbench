import pytest

from rl_world_ascii.providers.factory import ClientBuildConfig, build_client
from rl_world_ascii.providers.pricing import Pricing


@pytest.fixture
def pricing(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text(
        """
version: "test"
providers:
  vllm:
    "*":
      input_per_1m_tokens: 0.0
      output_per_1m_tokens: 0.0
      reasoning_per_1m_tokens: 0.0
  openai:
    gpt-4o-mini-test:
      input_per_1m_tokens: 0.15
      output_per_1m_tokens: 0.60
      reasoning_per_1m_tokens: null
"""
    )
    return Pricing.from_yaml(p)


def test_build_vllm_client(pricing):
    cfg = ClientBuildConfig(
        provider="vllm",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
    )
    client = build_client(cfg, pricing=pricing)
    assert client.provider == "vllm"
    assert client.model_id == "meta-llama/Llama-3.1-8B-Instruct"


def test_build_openai_client_requires_api_key(monkeypatch, pricing):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = ClientBuildConfig(provider="openai", model_id="gpt-4o-mini-test")
    client = build_client(cfg, pricing=pricing)
    assert client.provider == "openai"
    assert client.model_id == "gpt-4o-mini-test"


def test_build_unknown_provider_raises(pricing):
    cfg = ClientBuildConfig(provider="madeupprovider", model_id="x")
    with pytest.raises(ValueError, match="unknown provider"):
        build_client(cfg, pricing=pricing)
