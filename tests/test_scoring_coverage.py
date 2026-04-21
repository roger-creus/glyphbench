"""Tests for env-coverage visibility in GlyphBench scoring.

Covers:
  1. Full-coverage model produces no warnings, coverage = 1.0 per suite.
  2. Model missing >5% of a suite emits a warning and coverage < 0.95.
  3. Model runs an env not in baseline -> one warning (once per unique env),
     and that env is excluded from IQM rather than using 0.0 baseline.
  4. Baseline near zero logs a noisy-normalization note.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from eval.scoring import compute_glyphbench_scores


# ---------------------------------------------------------------------------
# Fixtures: small synthetic baseline + results tree
# ---------------------------------------------------------------------------


def _make_baseline(tmp_path: Path, envs: dict[str, float]) -> Path:
    """Write a minimal random_baseline.json with the given {env_id: mean_return}."""
    data = {
        env_id: {
            "env_id": env_id,
            "n_episodes": 1,
            "mean_return": mean,
            "std_return": 0.0,
            "min_return": mean,
            "max_return": mean,
            "median_return": mean,
            "mean_length": 10,
            "per_episode_returns": [mean],
        }
        for env_id, mean in envs.items()
    }
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(data))
    return p


def _write_model_results(
    results_dir: Path,
    model_name: str,
    harness: str,
    env_returns: dict[str, float],
) -> None:
    """Write per_env/*.json result files for one model + harness."""
    per_env = results_dir / model_name / harness / "per_env"
    per_env.mkdir(parents=True, exist_ok=True)
    for env_id, mean in env_returns.items():
        fname = env_id.replace("/", "_") + ".json"
        (per_env / fname).write_text(
            json.dumps({
                "env_id": env_id,
                "mean_return": mean,
                "parse_failure_rate": 0.0,
            }),
        )


def _make_baseline_suite(tmp_path: Path, suite_sizes: dict[str, int]) -> tuple[Path, dict[str, list[str]]]:
    """Build baseline with suite_sizes[suite] envs per suite, baseline=1.0 each.

    Returns (baseline_path, {suite: [env_id, ...]}).
    """
    envs: dict[str, float] = {}
    by_suite: dict[str, list[str]] = {}
    for suite, n in suite_sizes.items():
        by_suite[suite] = []
        for i in range(n):
            env_id = f"glyphbench/{suite}-env{i:02d}-v0"
            envs[env_id] = 1.0
            by_suite[suite].append(env_id)
    return _make_baseline(tmp_path, envs), by_suite


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_coverage_no_warnings_and_all_ones(tmp_path, caplog):
    """A model that ran every env -> no coverage warnings, coverage=1.0 per suite."""
    baseline_path, by_suite = _make_baseline_suite(
        tmp_path, {"atari": 10, "minigrid": 10},
    )
    results_dir = tmp_path / "results"
    all_envs = {e: 2.0 for suite_envs in by_suite.values() for e in suite_envs}
    _write_model_results(results_dir, "modelA", "markov_zeroshot", all_envs)

    with caplog.at_level(logging.WARNING, logger="eval.scoring"):
        scores = compute_glyphbench_scores(results_dir, baseline_path)

    entry = next(e for e in scores["leaderboard"] if e["model"] == "modelA")
    assert "coverage" in entry
    assert entry["coverage"]["atari"] == pytest.approx(1.0)
    assert entry["coverage"]["minigrid"] == pytest.approx(1.0)

    # No missing-envs coverage warnings
    coverage_warnings = [r for r in caplog.records if "missing" in r.getMessage().lower()]
    assert coverage_warnings == []


def test_partial_coverage_emits_warning(tmp_path, caplog):
    """Model missing 10% of atari envs -> warning, coverage['atari'] < 0.95."""
    baseline_path, by_suite = _make_baseline_suite(
        tmp_path, {"atari": 20, "minigrid": 10},
    )
    results_dir = tmp_path / "results"
    # Run only 18/20 atari envs (90%), full minigrid
    atari_subset = by_suite["atari"][:18]
    envs_run = {e: 2.0 for e in atari_subset + by_suite["minigrid"]}
    _write_model_results(results_dir, "modelB", "markov_zeroshot", envs_run)

    with caplog.at_level(logging.WARNING, logger="eval.scoring"):
        scores = compute_glyphbench_scores(results_dir, baseline_path)

    entry = next(e for e in scores["leaderboard"] if e["model"] == "modelB")
    assert entry["coverage"]["atari"] == pytest.approx(0.9)
    assert entry["coverage"]["minigrid"] == pytest.approx(1.0)
    assert entry["coverage"]["atari"] < 0.95

    msgs = [r.getMessage() for r in caplog.records]
    assert any("modelB" in m and "atari" in m and "missing" in m.lower() for m in msgs), (
        f"Expected coverage warning for modelB/atari, got: {msgs}"
    )


def test_env_not_in_baseline_is_skipped_with_one_warning(tmp_path, caplog):
    """An env the model ran but not in baseline -> skipped from IQM + one warning."""
    baseline_path, by_suite = _make_baseline_suite(tmp_path, {"atari": 8})
    results_dir = tmp_path / "results"

    # Model runs all 8 baseline envs plus one UNKNOWN env (not in baseline)
    unknown = "glyphbench/atari-ghostenv-v0"
    envs_run = {e: 2.0 for e in by_suite["atari"]}
    envs_run[unknown] = 9999.0  # Huge return; if used with baseline=0 it would dominate
    _write_model_results(results_dir, "modelC", "markov_zeroshot", envs_run)

    # Second model also runs the same unknown env -> warning must still be ONE
    _write_model_results(results_dir, "modelD", "markov_zeroshot", envs_run)

    with caplog.at_level(logging.WARNING, logger="eval.scoring"):
        scores = compute_glyphbench_scores(results_dir, baseline_path)

    # The unknown env should not appear in per_env rows
    per_env_ids = {r["env_id"] for r in scores["per_env"]}
    assert unknown not in per_env_ids

    # Exactly one warning for this unique missing env (across both models)
    msgs = [r.getMessage() for r in caplog.records if unknown in r.getMessage()]
    assert len(msgs) == 1, f"Expected exactly 1 warning for missing-from-baseline env, got {len(msgs)}: {msgs}"


def test_near_zero_baseline_logs_noisy_note(tmp_path, caplog):
    """Env with baseline ~= 0 -> logged as noisy normalization."""
    baseline_path = _make_baseline(
        tmp_path,
        {
            "glyphbench/atari-normal-v0": 10.0,
            "glyphbench/atari-zerob-v0": 0.0,
        },
    )
    results_dir = tmp_path / "results"
    _write_model_results(
        results_dir,
        "modelE",
        "markov_zeroshot",
        {
            "glyphbench/atari-normal-v0": 20.0,
            "glyphbench/atari-zerob-v0": 5.0,
        },
    )

    with caplog.at_level(logging.INFO, logger="eval.scoring"):
        compute_glyphbench_scores(results_dir, baseline_path)

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "zerob" in m and ("near-zero" in m.lower() or "noisy" in m.lower())
        for m in msgs
    ), f"Expected near-zero baseline note for zerob env, got: {msgs}"
