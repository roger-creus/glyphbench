"""Smoke + win/loss path tests for miniatari-tennis-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-tennis-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-tennis-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-tennis-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Action sequence captured from a "stand 1 cell left of ball" controller on
# seed=0 — the offset deflects the ball back at an angle the opponent
# under-tracks, so the agent wins three games (cumulative = +1.0).
_TENNIS_WIN_ACTIONS = (
    "LEFT NOOP NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT LEFT "
    "NOOP NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT NOOP "
    "NOOP LEFT LEFT LEFT LEFT LEFT LEFT NOOP NOOP NOOP RIGHT RIGHT RIGHT "
    "RIGHT RIGHT RIGHT NOOP NOOP LEFT LEFT LEFT LEFT LEFT LEFT NOOP NOOP "
    "NOOP LEFT LEFT LEFT NOOP NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT "
    "RIGHT RIGHT RIGHT RIGHT RIGHT LEFT NOOP NOOP NOOP RIGHT RIGHT RIGHT "
    "RIGHT RIGHT LEFT NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT LEFT "
    "NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT LEFT NOOP NOOP NOOP "
    "RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT NOOP NOOP LEFT LEFT LEFT LEFT LEFT"
).split()


def test_tennis_win_path() -> None:
    """seed=0 + a fixed offset-tracking action sequence wins 3 games.
    Asserts terminated, won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-tennis-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _TENNIS_WIN_ACTIONS:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_tennis_loss_path() -> None:
    """Stand at the left edge ignoring the ball: opp scores 3 games.
    Asserts terminated, won=False, cumulative >= -1.0."""
    env = make_env("glyphbench/miniatari-tennis-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(80):
        _, r, terminated, truncated, info = env.step(ai("LEFT"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path LEFT-spam should terminate via opp 3-game win"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9


def test_tennis_y_range_reaches_court_b() -> None:
    """Regression: with the y < COURT_B fix, the player can step DOWN to
    y == _COURT_B (the bottom playable row)."""
    env = make_env("glyphbench/miniatari-tennis-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    # Step DOWN until we hit the new max. Initial y = COURT_B - 1, so one
    # DOWN should reach y == COURT_B.
    for _ in range(3):
        env.step(ai("DOWN"))
    assert env._player_y == env._COURT_B
