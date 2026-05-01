"""Smoke + win/loss path tests for miniatari-choppercommand-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-choppercommand-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-choppercommand-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-choppercommand-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_choppercommand_has_trucks() -> None:
    """Identity-restoring change: env should expose 3 trucks at reset."""
    env = make_env("glyphbench/miniatari-choppercommand-v0")
    env.reset(seed=0)
    assert env._N_TRUCKS == 3
    assert len(env._trucks) == 3


def test_choppercommand_win_path() -> None:
    """Greedy chase + FIRE on seed=0 clears all 5 enemies before trucks die."""
    env = make_env("glyphbench/miniatari-choppercommand-v0")
    env.reset(seed=0)
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    up = env.action_spec.names.index("UP")
    down = env.action_spec.names.index("DOWN")
    fire = env.action_spec.names.index("FIRE")

    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        if not env._enemies:
            break
        same_row = [e for e in env._enemies if e[1] == env._player_y]
        if same_row:
            target = min(same_row, key=lambda e: abs(e[0] - env._player_x))
            facing_right = env._player_dir[0] > 0
            target_right = target[0] > env._player_x
            if target_right == facing_right:
                action = fire
            else:
                action = right if target_right else left
        else:
            target = min(
                env._enemies,
                key=lambda e: abs(e[1] - env._player_y) + abs(e[0] - env._player_x),
            )
            if env._player_y < target[1]:
                action = down
            elif env._player_y > target[1]:
                action = up
            elif env._player_x < target[0]:
                action = right
            else:
                action = left
        _, reward, terminated, _, info = env.step(action)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected greedy chase to terminate"
    assert won, f"expected win, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_choppercommand_loss_path() -> None:
    """Climb out of enemy reach, then NOOP: trucks get attacked one by one
    and the convoy is destroyed -> -1 terminal."""
    env = make_env("glyphbench/miniatari-choppercommand-v0")
    env.reset(seed=0)
    up = env.action_spec.names.index("UP")
    noop = env.action_spec.names.index("NOOP")

    cumulative = 0.0
    won = True
    terminated = False
    # First climb to top row to avoid enemy collision
    for _ in range(10):
        _, reward, terminated, _, info = env.step(up)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break

    # Then NOOP until convoy destroyed
    if not terminated:
        for _ in range(env.max_turns):
            _, reward, terminated, _, info = env.step(noop)
            cumulative += reward
            if terminated:
                won = bool(info.get("won"))
                break

    assert terminated, "expected NOOP-after-climb to lose convoy"
    assert not won, "loss path should not be a win"
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6
