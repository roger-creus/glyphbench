"""Smoke + win/loss path tests for miniatari-demonattack-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-demonattack-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-demonattack-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-demonattack-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_demonattack_win_path() -> None:
    """Track demons + FIRE on seed=3 clears all 5 before bombs hit."""
    env = make_env("glyphbench/miniatari-demonattack-v0")
    env.reset(seed=3)
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    fire = env.action_spec.names.index("FIRE")
    noop = env.action_spec.names.index("NOOP")

    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        if not env._demons:
            break
        in_col = any(d[0] == env._player_x for d in env._demons)
        if in_col:
            action = fire
        else:
            target = min(env._demons, key=lambda d: abs(d[0] - env._player_x))
            if env._player_x < target[0]:
                action = right
            elif env._player_x > target[0]:
                action = left
            else:
                action = fire
        _, reward, terminated, _, info = env.step(action)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected forced win path to terminate"
    assert won, f"expected win, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_demonattack_loss_path() -> None:
    """NOOP only on seed=0: bombs land on the cannon for terminal -1."""
    env = make_env("glyphbench/miniatari-demonattack-v0")
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
    assert terminated, "expected NOOP-only run to be hit by a bomb"
    assert not won, "loss path should not be a win"
    assert -1.0 - 1e-6 <= cumulative
