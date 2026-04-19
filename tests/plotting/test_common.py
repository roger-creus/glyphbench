"""Tests for plotting data loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _create_fake_run(tmp_path: Path, run_id: str, env_ids: list[str]) -> Path:
    """Create a minimal fake run directory with summary and turns parquet."""
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)

    summary_rows = []
    for env_id in env_ids:
        for seed in [0, 1]:
            summary_rows.append({
                "env_id": env_id, "seed": seed, "episode_idx": 0,
                "episode_return": float(seed + 1) * 0.5,
                "episode_length": 10 + seed, "terminated_reason": "terminated",
                "total_tokens_in": 100, "total_tokens_out": 20,
                "total_tokens_reasoning": 0, "total_dollar_cost": 0.0,
                "total_wall_time_s": 1.0, "mean_latency_s": 0.1,
                "p95_latency_s": 0.2, "action_parse_failure_rate": 0.0,
            })
    summary_df = pd.DataFrame(summary_rows)
    pq.write_table(pa.Table.from_pandas(summary_df), run_dir / "summary.parquet")

    turns_rows = []
    for env_id in env_ids:
        for seed in [0, 1]:
            for turn in range(10 + seed):
                turns_rows.append({
                    "env_id": env_id, "seed": seed, "episode_idx": 0,
                    "turn_index": turn, "wall_time_s": 0.1, "reward": 0.0,
                    "terminated": turn == 9 + seed, "truncated": False,
                    "action_index": 0, "action_name": "NOOP",
                    "action_parse_error": False, "action_parse_retries": 0,
                    "action_fell_back_to_noop": False,
                    "tokens_in": 10, "tokens_out": 2, "tokens_reasoning": 0,
                    "latency_provider_s": 0.1, "dollar_cost_turn": 0.0,
                })
    turns_df = pd.DataFrame(turns_rows)
    pq.write_table(pa.Table.from_pandas(turns_df), run_dir / "turns.parquet")
    return run_dir


class TestLoadRun:
    def test_load_run_returns_dict(self, tmp_path: Path) -> None:
        from atlas_rl.plotting.common import load_run
        run_dir = _create_fake_run(tmp_path, "run1", ["env-a"])
        result = load_run(str(run_dir))
        assert isinstance(result, dict)
        assert "summary" in result
        assert "turns" in result
        assert isinstance(result["summary"], pd.DataFrame)
        assert isinstance(result["turns"], pd.DataFrame)

    def test_load_run_summary_shape(self, tmp_path: Path) -> None:
        from atlas_rl.plotting.common import load_run
        run_dir = _create_fake_run(tmp_path, "run1", ["env-a", "env-b"])
        result = load_run(str(run_dir))
        assert len(result["summary"]) == 4  # 2 envs * 2 seeds

    def test_load_run_missing_dir_raises(self) -> None:
        from atlas_rl.plotting.common import load_run
        with pytest.raises(FileNotFoundError):
            load_run("/nonexistent/path")


class TestLoadRuns:
    def test_load_runs_concatenates(self, tmp_path: Path) -> None:
        from atlas_rl.plotting.common import load_runs
        _create_fake_run(tmp_path, "run1", ["env-a"])
        _create_fake_run(tmp_path, "run2", ["env-a"])
        result = load_runs([str(tmp_path / "run1"), str(tmp_path / "run2")])
        assert isinstance(result, pd.DataFrame)
        assert "run_id" in result.columns
        assert len(result) == 4  # 2 runs * 1 env * 2 seeds

    def test_load_runs_empty_list(self) -> None:
        from atlas_rl.plotting.common import load_runs
        result = load_runs([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestComputeNormalizedScores:
    def test_normalized_scores_basic(self, tmp_path: Path) -> None:
        from atlas_rl.plotting.common import compute_normalized_scores
        summary = pd.DataFrame({
            "env_id": ["env-a", "env-a", "env-b", "env-b"],
            "episode_return": [1.0, 2.0, 3.0, 4.0],
        })
        random_summary = pd.DataFrame({
            "env_id": ["env-a", "env-a", "env-b", "env-b"],
            "episode_return": [0.0, 0.0, 1.0, 1.0],
        })
        result = compute_normalized_scores(summary, random_summary)
        assert "normalized_return" in result.columns

    def test_normalized_scores_with_expert(self, tmp_path: Path) -> None:
        from atlas_rl.plotting.common import compute_normalized_scores
        summary = pd.DataFrame({
            "env_id": ["env-a", "env-a"],
            "episode_return": [5.0, 10.0],
        })
        random_summary = pd.DataFrame({
            "env_id": ["env-a", "env-a"],
            "episode_return": [0.0, 0.0],
        })
        expert = {"env-a": 20.0}
        result = compute_normalized_scores(summary, random_summary, expert_reference=expert)
        np.testing.assert_allclose(result["normalized_return"].values, [0.25, 0.5])
