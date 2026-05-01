"""Tests for the prime-rl rollout JSONL reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from glyphbench.rl.log_reader import load_rollouts, summary_by_env

SMOKE_DIR = Path("/home/roger/glyphbench/outputs/qwen35-4b-glyphbench-smoke-v1")


def _smoke_or_skip() -> Path:
    if not (SMOKE_DIR / "run_default" / "rollouts").exists():
        pytest.skip(f"smoke output not present at {SMOKE_DIR}")
    return SMOKE_DIR


def test_load_rollouts_smoke() -> None:
    out = _smoke_or_skip()
    df = load_rollouts(out)

    train = df[df["phase"] == "train"]
    eval_ = df[df["phase"] == "eval"]

    assert len(train) >= 80, f"expected >=80 train rows, got {len(train)}"
    assert len(eval_) >= 4, f"expected >=4 eval rows, got {len(eval_)}"
    assert train["env_id"].notna().all(), "env_id should be non-NaN for all train rows"
    assert set(train["step"].unique()) >= {0, 1, 2, 3, 4}

    # Bulky fields are dropped, key derived columns are present.
    for col in ("step", "phase", "env_id", "reward", "example_id"):
        assert col in df.columns
    # `info` is dropped after pulling out env_id/seed; bulky transcript
    # fields are dropped; `metrics` is retained for opt-in inspection.
    for col in ("prompt", "completion", "trajectory", "info"):
        assert col not in df.columns
    for col in ("env_id", "suite", "seed"):
        assert col in df.columns


def test_summary_by_env_smoke() -> None:
    out = _smoke_or_skip()
    df = load_rollouts(out)
    summary = summary_by_env(df)

    assert len(summary) > 1, "smoke run touched multiple envs"
    assert "mean_reward" in summary.columns
    assert "count" in summary.columns
    # Sorted by count descending.
    assert list(summary["count"]) == sorted(summary["count"], reverse=True)


def test_load_rollouts_synthetic(tmp_path: Path) -> None:
    rollouts_dir = tmp_path / "run_default" / "rollouts"
    step0 = rollouts_dir / "step_0"
    step1 = rollouts_dir / "step_1"
    step0.mkdir(parents=True)
    step1.mkdir(parents=True)

    train_rec = {
        "example_id": 1,
        "task": "default",
        "prompt": [{"role": "user", "content": "x" * 1000}],
        "completion": [{"role": "assistant", "content": "y" * 1000}],
        "info": {"env_id": "glyphbench/fake-env-v0", "seed": 1},
        "reward": 0.5,
        "episodic_return": 0.5,
        "episode_length": 7,
        "num_turns": 3,
        "forfeit_rate": 0.0,
        "xml_format_reward": 0.0,
        "is_completed": True,
        "is_truncated": False,
        "advantage": 0.1,
        "metrics": {"foo": 1.23},
    }
    eval_rec = {
        **train_rec,
        "example_id": 2,
        # Sometimes info comes in as a JSON-encoded string — exercise that path.
        "info": json.dumps({"env_id": "glyphbench/fake-env-v1", "seed": 2}),
        "reward": 1.0,
    }

    (step0 / "train_rollouts.jsonl").write_text(json.dumps(train_rec) + "\n")
    (step1 / "eval_rollouts.jsonl").write_text(json.dumps(eval_rec) + "\n")

    df = load_rollouts(tmp_path)
    assert len(df) == 2
    assert set(df["phase"]) == {"train", "eval"}
    assert set(df["step"]) == {0, 1}
    assert set(df["env_id"]) == {
        "glyphbench/fake-env-v0",
        "glyphbench/fake-env-v1",
    }
    for col in ("prompt", "completion", "info"):
        assert col not in df.columns

    summary = summary_by_env(df)
    assert len(summary) == 2
    assert {"mean_reward", "count", "completion_rate"} <= set(summary.columns)


def test_load_rollouts_missing_dir(tmp_path: Path) -> None:
    # Pointing at a path without run_default/rollouts should yield an empty df,
    # not crash.
    df = load_rollouts(tmp_path)
    assert df.empty


def test_load_rollouts_coalesces_legacy_parse_failure_rate(tmp_path: Path) -> None:
    """Legacy JSONL with parse_failure_rate (not forfeit_rate) should be
    coalesced into a forfeit_rate column by load_rollouts."""
    rollouts_dir = tmp_path / "run_default" / "rollouts" / "step_0"
    rollouts_dir.mkdir(parents=True)
    rec = {
        "example_id": 0,
        "info": {"env_id": "glyphbench/fake-env-v0", "seed": 0},
        "reward": 0.5,
        "is_completed": True,
        # Legacy key: no forfeit_rate, only parse_failure_rate
        "parse_failure_rate": 0.25,
    }
    (rollouts_dir / "train_rollouts.jsonl").write_text(json.dumps(rec) + "\n")
    df = load_rollouts(tmp_path)
    assert len(df) == 1
    assert "forfeit_rate" in df.columns
    assert df.iloc[0]["forfeit_rate"] == pytest.approx(0.25)


def test_load_rollouts_skips_bad_lines(tmp_path: Path) -> None:
    rollouts_dir = tmp_path / "run_default" / "rollouts" / "step_0"
    rollouts_dir.mkdir(parents=True)
    rec = {
        "example_id": 0,
        "info": {"env_id": "glyphbench/x"},
        "reward": 0.0,
        "is_completed": True,
    }
    (rollouts_dir / "train_rollouts.jsonl").write_text(
        "not-json\n" + json.dumps(rec) + "\n\n"
    )
    df = load_rollouts(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["env_id"] == "glyphbench/x"
