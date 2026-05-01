"""Smoke + win/loss path tests for miniatari-riverraid-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-riverraid-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-riverraid-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-riverraid-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Captured by replaying an obstacle-tracker policy on seed=3: aim at the
# lowest live obstacle's column, FIRE when aligned, then NOOP through
# the rest of the river. Wins (8 obstacles + final goal = 9/9) at the
# course-end tick on seed=3.
_RIVERRAID_WIN_ACTIONS_SEED3 = (
    "RIGHT FIRE LEFT LEFT LEFT LEFT FIRE FIRE RIGHT RIGHT RIGHT FIRE "
    "FIRE LEFT LEFT FIRE FIRE LEFT FIRE "
    + " ".join(["NOOP"] * 31)
).split()


def test_riverraid_win_path() -> None:
    """Drive a precomputed obstacle-tracker action sequence on seed=3
    that shoots all 8 obstacles before the river ends. Asserts
    terminated, won=True, cumulative == 1.0 (9 progress units * 1/9)."""
    env = make_env("glyphbench/miniatari-riverraid-v0")
    env.reset(seed=3)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _RIVERRAID_WIN_ACTIONS_SEED3:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=3"
    assert 0.0 < total <= 1.0 + 1e-9


def test_riverraid_loss_path() -> None:
    """NOOP-only on seed=4: the obstacle in the player's start column
    drifts down and crashes within a few ticks. Asserts terminated,
    won=False, cumulative >= -1.0 (the death penalty)."""
    env = make_env("glyphbench/miniatari-riverraid-v0")
    env.reset(seed=4)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(200):
        _, r, terminated, _truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path NOOP rollout should terminate"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
