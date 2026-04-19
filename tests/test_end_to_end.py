"""End-to-end foundation gate test.

This is the hard gate for Stage 0 completion. It runs the full benchmark runner
across all 5 pilot envs using MockLLMClient, then asserts the output is correct.

Spec reference: section 10.4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from atlas_rl.harness.mock_client import MockLLMClient
from atlas_rl.runner.config import RunConfig
from atlas_rl.runner.runner import run_benchmark

PILOT_ENV_IDS = [
    "atlas_rl/minigrid-empty-5x5-v0",
    "atlas_rl/minihack-room-5x5-v0",
    "atlas_rl/procgen-coinrun-v0",
    "atlas_rl/atari-pong-v0",
    "atlas_rl/craftax-classic-v0",
]

SEEDS = [0, 1]
EPISODES_PER_ENV = 1
MAX_TURNS = 50


@pytest.fixture
def gate_config(tmp_path: Path) -> RunConfig:
    """Build a RunConfig for the foundation gate test."""
    return RunConfig(
        run_id="foundation-gate-test",
        provider="mock",
        model_id="mock-model",
        seeds=SEEDS,
        episodes_per_env=EPISODES_PER_ENV,
        max_turns_per_episode=MAX_TURNS,
        concurrency=1,
        budget_usd=None,
        envs=PILOT_ENV_IDS,
        output_dir=str(tmp_path),
        pricing_yaml="pricing.yaml",
        dashboard=False,
    )


@pytest.mark.asyncio
async def test_foundation_gate(gate_config: RunConfig, tmp_path: Path) -> None:
    """The hard gate test for Stage 0 completion."""
    import atlas_rl  # noqa: F401 -- trigger env registration

    mock_response = json.dumps({"action": "MOVE_FORWARD"})

    def client_factory() -> MockLLMClient:
        return MockLLMClient.always(mock_response)

    # Run the benchmark -- should not raise
    await run_benchmark(gate_config, client_factory=client_factory)

    # Check output directory exists
    run_dir = tmp_path / "foundation-gate-test"
    assert run_dir.exists(), f"Run directory not created: {run_dir}"

    # Check summary.parquet exists and has correct row count
    summary_path = run_dir / "summary.parquet"
    assert summary_path.exists(), "summary.parquet not created"
    summary_df = pq.read_table(summary_path).to_pandas()

    expected_rows = len(PILOT_ENV_IDS) * len(SEEDS) * EPISODES_PER_ENV
    assert len(summary_df) == expected_rows, (
        f"Expected {expected_rows} summary rows, got {len(summary_df)}"
    )

    # All env_ids should be present
    assert set(summary_df["env_id"].unique()) == set(PILOT_ENV_IDS)

    # Check turns.parquet exists and has rows
    turns_path = run_dir / "turns.parquet"
    assert turns_path.exists(), "turns.parquet not created"
    turns_df = pq.read_table(turns_path).to_pandas()
    assert len(turns_df) > 0, "turns.parquet is empty"

    # Every episode in summary should have corresponding turns
    for _, row in summary_df.iterrows():
        ep_turns = turns_df[
            (turns_df["env_id"] == row["env_id"])
            & (turns_df["seed"] == row["seed"])
            & (turns_df["episode_idx"] == row["episode_idx"])
        ]
        assert len(ep_turns) > 0, (
            f"No turns for {row['env_id']} seed={row['seed']} ep={row['episode_idx']}"
        )
        assert len(ep_turns) == row["episode_length"], (
            f"Turn count mismatch for {row['env_id']}: "
            f"expected {row['episode_length']}, got {len(ep_turns)}"
        )

    # Check summary has valid numeric columns
    for col in [
        "episode_return", "episode_length", "total_tokens_in",
        "total_tokens_out", "total_dollar_cost", "mean_latency_s",
    ]:
        assert col in summary_df.columns, f"Missing column: {col}"
        assert summary_df[col].notna().all(), f"NaN values in column: {col}"

    # Pricing should return 0.0 for mock provider
    assert (summary_df["total_dollar_cost"] == 0.0).all()

    # Normalized scoring should be computable
    from atlas_rl.plotting.common import compute_normalized_scores
    normalized = compute_normalized_scores(summary_df, summary_df)
    assert "normalized_return" in normalized.columns
    assert normalized["normalized_return"].notna().all()
