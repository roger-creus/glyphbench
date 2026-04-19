from pathlib import Path

import pytest

from rl_world_ascii.providers.pricing import Pricing


@pytest.fixture
def pricing_file(tmp_path: Path) -> Path:
    p = tmp_path / "pricing.yaml"
    p.write_text(
        """
version: "test-v1"
providers:
  openai:
    gpt-4o-test:
      input_per_1m_tokens: 2.00
      output_per_1m_tokens: 10.00
      reasoning_per_1m_tokens: null
  vllm:
    "*":
      input_per_1m_tokens: 0.0
      output_per_1m_tokens: 0.0
      reasoning_per_1m_tokens: 0.0
"""
    )
    return p


def test_load_pricing_from_file(pricing_file: Path):
    p = Pricing.from_yaml(pricing_file)
    assert p.version == "test-v1"


def test_compute_cost_for_known_model(pricing_file: Path):
    p = Pricing.from_yaml(pricing_file)
    # 1M input @ $2, 0.5M output @ $10 -> $2 + $5 = $7
    cost = p.compute_cost(
        provider="openai",
        model_id="gpt-4o-test",
        tokens_in=1_000_000,
        tokens_out=500_000,
        tokens_reasoning=0,
    )
    assert cost == pytest.approx(7.0)


def test_compute_cost_for_vllm_wildcard_is_zero(pricing_file: Path):
    p = Pricing.from_yaml(pricing_file)
    cost = p.compute_cost(
        provider="vllm",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        tokens_in=999_999,
        tokens_out=999_999,
        tokens_reasoning=0,
    )
    assert cost == 0.0


def test_compute_cost_for_unknown_model_returns_none(pricing_file: Path):
    p = Pricing.from_yaml(pricing_file)
    cost = p.compute_cost(
        provider="openai",
        model_id="unknown-model-9000",
        tokens_in=100,
        tokens_out=100,
        tokens_reasoning=0,
    )
    assert cost is None


def test_compute_cost_with_reasoning_null_counts_as_output(pricing_file: Path):
    # reasoning_per_1m_tokens null: reasoning tokens should contribute at output rate
    p = Pricing.from_yaml(pricing_file)
    cost = p.compute_cost(
        provider="openai",
        model_id="gpt-4o-test",
        tokens_in=0,
        tokens_out=0,
        tokens_reasoning=100_000,  # 0.1M * $10 = $1
    )
    assert cost == pytest.approx(1.0)
