"""Tests for the custom GRPO + per-env σ advantage function."""

from __future__ import annotations

import math

import pytest

from glyphbench.rl.advantage import (
    GlyphbenchAdvantageState,
    compute_advantages_with_env_norm,
)


def make_rollout(env_name: str, example_id: int, reward: float) -> dict:
    """Minimal rollout shape — only the keys our advantage function reads."""
    return {
        "env_name": env_name,
        "example_id": example_id,
        "reward": reward,
        "metrics": {},  # advantage may set keys here
    }


class TestSingleEnvSingleGroup:
    def test_centers_advantages_to_zero_mean(self) -> None:
        rollouts = [
            make_rollout("env-a", example_id=1, reward=1.0),
            make_rollout("env-a", example_id=1, reward=2.0),
            make_rollout("env-a", example_id=1, reward=3.0),
            make_rollout("env-a", example_id=1, reward=4.0),
        ]
        state = GlyphbenchAdvantageState(sigma_min=1e-6)
        compute_advantages_with_env_norm(rollouts, samples_per_problem=4, state=state)
        # Group mean is 2.5; raw deviations [-1.5, -0.5, 0.5, 1.5].
        # σ from welford on these four observations = std([1,2,3,4], ddof=1) = sqrt(5/3) ≈ 1.291.
        sigma = math.sqrt(5.0 / 3.0)
        expected = [(r - 2.5) / sigma for r in (1.0, 2.0, 3.0, 4.0)]
        actual = [r["advantage"] for r in rollouts]
        for a, e in zip(actual, expected):
            assert a == pytest.approx(e, rel=1e-9)

    def test_zero_variance_group_yields_zero_advantage(self) -> None:
        rollouts = [
            make_rollout("env-a", example_id=1, reward=5.0),
            make_rollout("env-a", example_id=1, reward=5.0),
        ]
        state = GlyphbenchAdvantageState(sigma_min=0.1)
        compute_advantages_with_env_norm(rollouts, samples_per_problem=2, state=state)
        # All rewards equal -> deviation 0 -> advantage 0 regardless of σ.
        assert rollouts[0]["advantage"] == pytest.approx(0.0)
        assert rollouts[1]["advantage"] == pytest.approx(0.0)


class TestMultipleEnvsMixedBatch:
    def test_envs_get_independent_sigmas(self) -> None:
        # env-a: rewards [0.0, 1.0]  -> σ_a = sqrt(0.5) ≈ 0.707
        # env-b: rewards [10.0, 20.0] -> σ_b = sqrt(50) ≈ 7.07
        rollouts = [
            make_rollout("env-a", example_id=1, reward=0.0),
            make_rollout("env-a", example_id=1, reward=1.0),
            make_rollout("env-b", example_id=2, reward=10.0),
            make_rollout("env-b", example_id=2, reward=20.0),
        ]
        state = GlyphbenchAdvantageState(sigma_min=1e-9)
        compute_advantages_with_env_norm(rollouts, samples_per_problem=2, state=state)
        # env-a rollouts: deviations [-0.5, 0.5] / 0.707...
        # env-b rollouts: deviations [-5.0, 5.0] / 7.07...
        # Both groups should have advantages of equivalent magnitude after σ-normalize.
        a0, a1, b0, b1 = (r["advantage"] for r in rollouts)
        assert a0 == pytest.approx(-a1, rel=1e-9)
        assert b0 == pytest.approx(-b1, rel=1e-9)
        # |a| ≈ |b| because per-env σ cancelled the magnitude difference
        assert abs(a0) == pytest.approx(abs(b0), rel=1e-3)


class TestSigmaPersistsAcrossBatches:
    def test_repeated_calls_keep_growing_n(self) -> None:
        state = GlyphbenchAdvantageState(sigma_min=1e-9)
        for _ in range(3):
            rollouts = [
                make_rollout("env-a", example_id=1, reward=1.0),
                make_rollout("env-a", example_id=1, reward=2.0),
            ]
            compute_advantages_with_env_norm(rollouts, samples_per_problem=2, state=state)
        # 3 batches × 2 rollouts each = 6 observations
        assert state.welford.n("env-a") == 6


class TestRolloutMetricsAttached:
    def test_per_rollout_welford_metrics_written(self) -> None:
        rollouts = [
            make_rollout("env-a", example_id=1, reward=1.0),
            make_rollout("env-a", example_id=1, reward=2.0),
        ]
        state = GlyphbenchAdvantageState(sigma_min=1e-9)
        compute_advantages_with_env_norm(rollouts, samples_per_problem=2, state=state)
        for r in rollouts:
            assert "welford_env_mean" in r["metrics"]
            assert "welford_env_std" in r["metrics"]
            assert "welford_env_n" in r["metrics"]
