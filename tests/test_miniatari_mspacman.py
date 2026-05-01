"""Smoke + win/loss path tests for miniatari-mspacman-v0."""
from __future__ import annotations

from collections import deque

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-mspacman-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-mspacman-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-mspacman-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Hand-derived from a greedy planner (BFS to nearest dot, ghost-cell aware)
# on seed=0. With the post-fix behaviour the planner clears every corridor
# dot in 98 ticks. The sequence is deterministic given the fixed maze and
# scripted ghost AI.
_MSPACMAN_WIN_ACTIONS_SEED0 = (
    "RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT "
    "UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP "
    "RIGHT RIGHT RIGHT UP UP LEFT LEFT LEFT UP UP RIGHT RIGHT RIGHT "
    "RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN LEFT "
    "LEFT RIGHT RIGHT DOWN DOWN LEFT LEFT LEFT UP DOWN DOWN DOWN "
    "RIGHT RIGHT RIGHT UP UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT "
    "LEFT LEFT UP DOWN DOWN DOWN DOWN UP LEFT LEFT LEFT UP UP UP UP "
    "RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN"
).split()


def test_mspacman_win_path() -> None:
    """Drive the agent through a precomputed winning action sequence on
    seed=0. Asserts terminated, won=True, and 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-mspacman-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _MSPACMAN_WIN_ACTIONS_SEED0:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_mspacman_loss_path() -> None:
    """Stand still. After the pen-exit phase the ghosts march out of the
    pen and chase the player; one will reach (1,8) within ~30 ticks.
    Asserts terminated, won=False, cumulative >= -1.0 (the death penalty
    dominates any small dot bonus the agent would not have earned here)."""
    env = make_env("glyphbench/miniatari-mspacman-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(80):
        _, r, terminated, _truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path script should reach terminal state"
    assert not won, "loss-path script should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9


def test_mspacman_dots_cover_all_corridors() -> None:
    """Regression: every open corridor cell except the ghost pen and the
    player start should hold a dot at reset (not just the top 2 rows)."""
    env = make_env("glyphbench/miniatari-mspacman-v0")
    env.reset(seed=0)
    # Bottom row (row 8) must contain dots — used to be empty.
    assert any((x, 8) in env._dots for x in range(2, 13))
    # Row 7 (open corridor) must contain dots.
    assert any((x, 7) in env._dots for x in range(2, 13))
    # Row 5 corridor cells (cols 1..4 and 9..12) must contain dots.
    assert any((x, 5) in env._dots for x in [1, 2, 3, 4, 9, 10, 11, 12])
