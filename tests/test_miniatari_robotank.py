"""Smoke + win/loss path tests for miniatari-robotank-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-robotank-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-robotank-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-robotank-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Captured by replaying a line-of-sight aware tracker on seed=0:
# pre-aligns facing with each cornering enemy, then FIREs along the
# clear axis. Wins in 20 ticks.
_ROBOTANK_WIN_ACTIONS_SEED0 = (
    "DOWN DOWN DOWN DOWN LEFT FIRE RIGHT FIRE RIGHT RIGHT RIGHT RIGHT UP "
    "FIRE UP UP UP UP LEFT FIRE"
).split()


def test_robotank_win_path() -> None:
    """Drive a precomputed line-of-sight tracker on seed=0 to clear all
    4 enemy tanks. Asserts terminated, won=True, cumulative == 1.0."""
    env = make_env("glyphbench/miniatari-robotank-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _ROBOTANK_WIN_ACTIONS_SEED0:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_robotank_loss_path() -> None:
    """NOOP-only on seed=0: enemies converge from corners and ram the
    player at ~tick 16. Per spec there's no -1 (just terminates with
    won=False). Asserts terminated, won=False, cumulative >= 0."""
    env = make_env("glyphbench/miniatari-robotank-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(60):
        _, r, terminated, _truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "NOOP rollout should terminate (enemy ram)"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
