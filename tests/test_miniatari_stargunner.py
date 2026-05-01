"""Smoke + win/loss path tests for miniatari-stargunner-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-stargunner-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-stargunner-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-stargunner-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_stargunner_win_path() -> None:
    """Force the agent through scripted actions to clear all 5 enemies on
    seed=0. Asserts terminated, won=True, and 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-stargunner-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    # seed=0 spawns enemies at (12,8), (11,5), (10,3) with player at (1,5).
    # The script kills the visible 3 then waits for the next 2 spawns.
    actions = (
        ["FIRE", "UP", "UP", "FIRE", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "FIRE"]
        + ["UP"] * 8 + ["FIRE"]
        + ["NOOP"] * 5 + ["FIRE"]
        + ["NOOP"] * 30 + ["FIRE"] * 5
    )
    total = 0.0
    terminated = False
    won = False
    for name in actions:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_stargunner_loss_path() -> None:
    """NOOP only: enemies march in and breach the player zone.
    Asserts terminated, won=False, cumulative >= -1.0 (no death penalty)."""
    env = make_env("glyphbench/miniatari-stargunner-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(60):
        _, r, terminated, truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path NOOP rollout should terminate via breach"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
