"""Smoke + win/loss path tests for miniatari-boxing-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-boxing-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-boxing-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-boxing-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_boxing_win_path() -> None:
    """Greedy chase + PUNCH on seed=0 reaches 5 hits before opponent does."""
    env = make_env("glyphbench/miniatari-boxing-v0")
    env.reset(seed=0)
    up = env.action_spec.names.index("UP")
    down = env.action_spec.names.index("DOWN")
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    punch = env.action_spec.names.index("PUNCH")

    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        d = abs(env._player_x - env._opp_x) + abs(env._player_y - env._opp_y)
        if d <= 2:
            action = punch
        elif env._player_x < env._opp_x:
            action = right
        elif env._player_x > env._opp_x:
            action = left
        elif env._player_y < env._opp_y:
            action = down
        elif env._player_y > env._opp_y:
            action = up
        else:
            action = punch
        _, reward, terminated, _, info = env.step(action)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected greedy chase to terminate"
    assert won, f"expected win, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_boxing_loss_path() -> None:
    """NOOP only on seed=0: opponent eventually corners + lands 5 punches."""
    env = make_env("glyphbench/miniatari-boxing-v0")
    env.reset(seed=0)
    noop = env.action_spec.names.index("NOOP")

    cumulative = 0.0
    won = True
    terminated = False
    for _ in range(env.max_turns):
        _, reward, terminated, _, info = env.step(noop)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected NOOP-only loss to terminate"
    assert not won, f"loss path should set won=False, got won={won}"
    assert -1.0 - 1e-6 <= cumulative
