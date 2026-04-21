"""Smoke test for run_benchmark: mock client + DummyEnv end-to-end."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from glyphbench.harness.mock_client import MockLLMClient, ScriptedResponse
from glyphbench.runner.config import RunConfig
from glyphbench.runner.runner import run_benchmark


@pytest.fixture
def smoke_config(tmp_path: Path) -> RunConfig:
    # Copy pricing.yaml into tmp so the runner can load it without touching the repo root.
    pricing_src = Path("pricing.yaml")
    pricing_dst = tmp_path / "pricing.yaml"
    if pricing_src.exists():
        shutil.copy(pricing_src, pricing_dst)
    return RunConfig(
        run_id="unit-smoke",
        provider="mock",
        model_id="mock-model",
        seeds=[0],
        episodes_per_env=1,
        max_turns_per_episode=10,
        concurrency=1,
        envs=["glyphbench/__dummy-v0"],
        output_dir=str(tmp_path / "runs"),
        pricing_yaml=str(pricing_dst),
        dashboard=False,
    )


async def test_runner_produces_parquet_with_mock_client(smoke_config: RunConfig) -> None:
    import glyphbench  # noqa: F401 — triggers env registration

    def client_factory() -> MockLLMClient:
        return MockLLMClient(
            scripted=[
                ScriptedResponse(text='{"action": "EAST"}'),
                ScriptedResponse(text='{"action": "EAST"}'),
                ScriptedResponse(text='{"action": "SOUTH"}'),
                ScriptedResponse(text='{"action": "SOUTH"}'),
            ]
            * 10  # plenty of scripted responses
        )

    await run_benchmark(smoke_config, client_factory=client_factory)

    summary = pd.read_parquet(
        Path(smoke_config.output_dir) / "unit-smoke" / "summary.parquet"
    )
    turns = pd.read_parquet(
        Path(smoke_config.output_dir) / "unit-smoke" / "turns.parquet"
    )
    assert len(summary) == 1  # 1 env x 1 seed x 1 episode
    assert len(turns) >= 1
    # DummyEnv: east east south south reaches goal; episode_return should be 1.0
    assert summary["episode_return"].iloc[0] == 1.0
