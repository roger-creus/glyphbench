"""Smoke + win/loss path tests for miniatari-defender-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-defender-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-defender-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-defender-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_defender_win_path() -> None:
    """Greedy rescue (descend to nearest human, climb back to row 0,
    repeat) on seed=0 saves all 3 humans."""
    env = make_env("glyphbench/miniatari-defender-v0")
    env.reset(seed=0)
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    up = env.action_spec.names.index("UP")
    down = env.action_spec.names.index("DOWN")
    noop = env.action_spec.names.index("NOOP")

    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        if env._carrying:
            action = up if env._player_y > 0 else noop
        else:
            if not env._humans:
                break
            target_x = min(env._humans, key=lambda x: abs(x - env._player_x))
            target_y = env._GROUND_Y
            if env._player_x == target_x:
                action = down if env._player_y < target_y else noop
            else:
                action = right if env._player_x < target_x else left
        _, reward, terminated, _, info = env.step(action)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected greedy rescue to terminate"
    assert won, f"expected win, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_defender_loss_path() -> None:
    """seed=2 spawns a lander aligned with a human column; NOOP-only
    lets the lander reach the human and abduct."""
    env = make_env("glyphbench/miniatari-defender-v0")
    env.reset(seed=2)
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
    assert terminated, "expected NOOP-only run on aligned seed to terminate"
    assert not won, "loss path should not be a win"
    # Defender's loss path emits no -1; cumulative is 0.
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6
