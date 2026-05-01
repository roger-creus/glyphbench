"""Smoke + win/loss path tests for miniatari-timepilot-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-timepilot-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-timepilot-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-timepilot-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Action sequence captured from a corner-camp + line-of-sight FIRE
# strategy on seed=3: walk to (1, 10) then turn-and-shoot whichever
# converging enemy enters firing range.
_TIMEPILOT_WIN_ACTIONS = [
    "LEFT", "LEFT", "LEFT", "LEFT", "LEFT", "LEFT",
    "DOWN", "DOWN", "DOWN", "DOWN",
    "NOOP", "NOOP",
    "UP", "FIRE",
    "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
    "FIRE",
    "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
    "FIRE",
    "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
    "FIRE",
    "NOOP",
    "FIRE",
]


def test_timepilot_win_path() -> None:
    """seed=3 + a corner-camp action sequence kills all 5 enemies.
    Asserts terminated, won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-timepilot-v0")
    env.reset(seed=3)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _TIMEPILOT_WIN_ACTIONS:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=3"
    assert 0.0 < total <= 1.0 + 1e-9


def test_timepilot_loss_path() -> None:
    """NOOP only on seed=0: a converging enemy rams the player.
    Asserts terminated, won=False, cumulative >= -1.0 (no death penalty)."""
    env = make_env("glyphbench/miniatari-timepilot-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(50):
        _, r, terminated, truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path NOOP rollout should terminate via collision"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
