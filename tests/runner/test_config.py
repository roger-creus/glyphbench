import pytest
import yaml

from rl_world_ascii.runner.config import HarnessConfig, RunConfig


def _valid_dict():
    return {
        "run_id": "test-run",
        "provider": "vllm",
        "model_id": "llama-3.1-8b",
        "base_url": "http://localhost:8000/v1",
        "seeds": [0, 1, 2],
        "envs": ["rl_world_ascii/__dummy-v0"],
    }


def test_minimum_valid_config():
    cfg = RunConfig(**_valid_dict())
    assert cfg.run_id == "test-run"
    assert cfg.temperature == 0.7  # default
    assert cfg.concurrency == 4  # default
    assert cfg.episodes_per_env == 10  # default
    assert cfg.budget_usd is None
    assert isinstance(cfg.harness, HarnessConfig)
    assert cfg.harness.retry_malformed_n == 3


def test_vllm_requires_base_url():
    data = _valid_dict()
    data.pop("base_url")
    with pytest.raises(ValueError, match="base_url"):
        RunConfig(**data)


def test_non_vllm_does_not_require_base_url():
    data = _valid_dict()
    data["provider"] = "openai"
    data["model_id"] = "gpt-4o-mini-2024-07-18"
    data.pop("base_url")
    cfg = RunConfig(**data)
    assert cfg.base_url is None


def test_load_from_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(_valid_dict()))
    cfg = RunConfig.from_yaml(path)
    assert cfg.run_id == "test-run"
