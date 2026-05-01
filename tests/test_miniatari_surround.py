"""Smoke + win/loss path tests for miniatari-surround-v0."""
from __future__ import annotations

import numpy as np

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-surround-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-surround-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-surround-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_surround_win_path() -> None:
    """seed=14 + a tight box-spiral lures the opponent into its own trail
    and crashes it. Asserts terminated, won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-surround-v0")
    env.reset(seed=14)
    ai = env.action_spec.names.index
    plan = (
        ["UP"] * 4 + ["RIGHT"] * 5 + ["DOWN"] * 4 + ["LEFT"] * 5
        + ["UP"] * 3 + ["RIGHT"] * 4
    )
    total = 0.0
    terminated = False
    won = False
    for name in plan:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=14"
    assert 0.0 < total <= 1.0 + 1e-9


def test_surround_loss_or_bound_path() -> None:
    """surround has no scriptable loss (opp randomness). Random rollouts
    must always satisfy the [-1, 1] cumulative-return bound. We run 10
    seeds of pure random play and assert the bound holds and that at
    least one terminates within 200 steps."""
    rng = np.random.default_rng(0)
    any_terminated = False
    for seed in range(10):
        env = make_env("glyphbench/miniatari-surround-v0")
        env.reset(seed=seed)
        total = 0.0
        terminated = False
        for _ in range(200):
            action = int(rng.integers(env.action_spec.n))
            _, r, terminated, truncated, info = env.step(action)
            total += r
            if terminated or truncated:
                any_terminated = any_terminated or terminated
                break
        assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9, (
            f"seed={seed} cumulative {total} outside [-1, 1]"
        )
    assert any_terminated, "no random rollout terminated; surround is too easy"
