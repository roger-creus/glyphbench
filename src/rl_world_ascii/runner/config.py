"""Pydantic run-config model loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class HarnessConfig(BaseModel):
    retry_malformed_n: int = 3
    recent_actions_window: int = 5
    trajectory_logging: bool = False


class RunConfig(BaseModel):
    run_id: str
    provider: str                      # "vllm" | "openai" | "anthropic" | "gemini" | "mock"
    model_id: str
    base_url: str | None = None
    temperature: float = 0.7
    max_output_tokens: int = 2048
    model_seed: int | None = None
    seeds: list[int]
    episodes_per_env: int = 10
    max_turns_per_episode: int = 500
    budget_usd: float | None = None
    concurrency: int = 4
    envs: list[str]
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    output_dir: str = "runs/"
    pricing_yaml: str = "pricing.yaml"
    dashboard: bool = True

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        if self.provider == "vllm" and not self.base_url:
            raise ValueError("vllm provider requires base_url")
        if self.episodes_per_env <= 0:
            raise ValueError("episodes_per_env must be > 0")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not self.envs:
            raise ValueError("envs must be non-empty")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
