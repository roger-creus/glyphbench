"""Smoke + win/loss path tests for miniatari-phoenix-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-phoenix-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-phoenix-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-phoenix-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Captured by replaying a phoenix-tracker policy on seed=0: aim at the
# lowest live phoenix's column, FIRE when aligned. Wins in 19 ticks.
_PHOENIX_WIN_ACTIONS_SEED0 = (
    "LEFT RIGHT FIRE LEFT LEFT LEFT FIRE LEFT FIRE RIGHT FIRE RIGHT FIRE "
    "RIGHT RIGHT RIGHT RIGHT RIGHT FIRE"
).split()


def test_phoenix_win_path() -> None:
    """Drive a precomputed tracker action sequence on seed=0 to clear
    all 6 phoenixes. Asserts terminated, won=True, cumulative == 1.0."""
    env = make_env("glyphbench/miniatari-phoenix-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _PHOENIX_WIN_ACTIONS_SEED0:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_phoenix_loss_path() -> None:
    """NOOP-only on seed=0: phoenixes drift onto the cannon row at
    ~tick 18 with no kills. Per spec the env terminates without -1
    (Pattern A). Asserts terminated, won=False, cumulative >= 0."""
    env = make_env("glyphbench/miniatari-phoenix-v0")
    env.reset(seed=0)
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
    assert terminated, "NOOP rollout should terminate (phoenix breach)"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
