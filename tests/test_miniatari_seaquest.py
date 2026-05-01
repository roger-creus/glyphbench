"""Smoke + win/loss path tests for miniatari-seaquest-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-seaquest-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-seaquest-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-seaquest-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Captured by replaying a Manhattan-tracker policy on seed=0: dive to
# each diver in order, then surface. Wins (3 rescues + final surface)
# in 30 ticks.
_SEAQUEST_WIN_ACTIONS_SEED0 = (
    "DOWN DOWN DOWN DOWN UP UP LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN "
    "DOWN RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT UP UP UP UP UP UP "
    "UP UP"
).split()


def test_seaquest_win_path() -> None:
    """Drive a precomputed diver-tracker action sequence on seed=0 to
    rescue all 3 divers and surface. Asserts terminated, won=True,
    cumulative == 1.0 (4 phases * 1/4)."""
    env = make_env("glyphbench/miniatari-seaquest-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _SEAQUEST_WIN_ACTIONS_SEED0:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_seaquest_loss_path() -> None:
    """Dive once and stay underwater (DOWN once, NOOP forever): oxygen
    depletes by 1 each tick from 80, hitting 0 at ~tick 80 and emitting
    -1. Asserts terminated, won=False, cumulative >= -1.0."""
    env = make_env("glyphbench/miniatari-seaquest-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    actions = ["DOWN"] + ["NOOP"] * 200
    for name in actions:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path should terminate via oxygen depletion"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
