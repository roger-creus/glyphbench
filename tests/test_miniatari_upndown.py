"""Smoke + win/loss path tests for miniatari-upndown-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-upndown-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-upndown-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-upndown-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Action sequence captured from a nearest-flag chaser with reactive JUMP
# on seed=1: hits all 5 flags before any car hits the player.
_UPNDOWN_WIN_ACTIONS = [
    "UP", "UP", "RIGHT", "RIGHT", "UP", "UP", "UP", "LEFT", "LEFT",
    "UP", "UP", "LEFT", "UP", "LEFT", "DOWN", "DOWN",
    "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT",
]


def test_upndown_win_path() -> None:
    """seed=1 + a captured action sequence collects all 5 flags.
    Asserts terminated, won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-upndown-v0")
    env.reset(seed=1)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _UPNDOWN_WIN_ACTIONS:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=1"
    assert 0.0 < total <= 1.0 + 1e-9


def test_upndown_loss_path() -> None:
    """NOOP only on seed=1: a car eventually drifts onto the player.
    Asserts terminated, won=False, cumulative >= -1.0 (no death penalty)."""
    env = make_env("glyphbench/miniatari-upndown-v0")
    env.reset(seed=1)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(150):
        _, r, terminated, truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path NOOP rollout should terminate via car hit"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
