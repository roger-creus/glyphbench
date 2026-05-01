"""Smoke + win/loss path tests for miniatari-centipede-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-centipede-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-centipede-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-centipede-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_centipede_win_path() -> None:
    """Greedy chase + FIRE clears all 8 segments on seed=0."""
    env = make_env("glyphbench/miniatari-centipede-v0")
    env.reset(seed=0)
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    fire = env.action_spec.names.index("FIRE")
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        if not env._segments:
            break
        # If a segment is in our column AND fire is ready: shoot.
        has_segment = any(s[0] == env._player_x for s in env._segments)
        if has_segment and env._fire_cd == 0:
            action = fire
        else:
            # Track lowest segment's column.
            target_x = min(env._segments, key=lambda s: (s[1], abs(s[0] - env._player_x)))[0]
            if env._player_x < target_x:
                action = right
            elif env._player_x > target_x:
                action = left
            else:
                action = noop
        _, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected greedy chase to terminate"
    assert won, f"expected win, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_centipede_loss_path() -> None:
    """NOOP only: centipede slithers down and head reaches player row."""
    env = make_env("glyphbench/miniatari-centipede-v0")
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
    assert terminated, "expected NOOP-only run to terminate"
    assert not won, "loss path should not be a win"
    # Centipede has no -1 (Pattern A); just terminates with whatever
    # progress was earned.
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6
