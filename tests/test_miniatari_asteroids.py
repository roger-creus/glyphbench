"""Smoke + win/loss path tests for miniatari-asteroids-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-asteroids-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-asteroids-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-asteroids-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Scripted move-and-fire path that destroys all 5 rocks on seed=15
# (16 steps). Asteroids on seed=0 spawn straight on the player's
# columns/rows and crowd quickly; seed=15 has a calmer initial layout
# in which the pinned sequence below cleanly clears the field.
_ASTEROIDS_WIN_SEQUENCE_SEED15 = [
    "LEFT", "UP", "FIRE", "LEFT", "FIRE", "UP", "RIGHT", "DOWN",
    "RIGHT", "FIRE", "RIGHT", "FIRE", "RIGHT", "RIGHT", "RIGHT",
    "FIRE",
]


def test_asteroids_win_path() -> None:
    """Scripted clear of 5 asteroids on seed=15."""
    env = make_env("glyphbench/miniatari-asteroids-v0")
    env.reset(seed=15)
    cumulative = 0.0
    won = False
    terminated = False
    info: dict = {}
    for name in _ASTEROIDS_WIN_SEQUENCE_SEED15:
        action = env.action_spec.names.index(name)
        _, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "win sequence must terminate the episode"
    assert won, "win sequence must set won=True"
    assert 0.0 < cumulative <= 1.0 + 1e-6, (
        f"cumulative {cumulative} not in (0, 1]"
    )


def test_asteroids_loss_path() -> None:
    """NOOP-only lets a drifting rock collide with the ship.

    Asteroids uses Pattern A semantics on death (no -1 penalty),
    so the loss-path contract is cumulative >= -1 and won=False.
    """
    env = make_env("glyphbench/miniatari-asteroids-v0")
    env.reset(seed=0)
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = True
    terminated = False
    info: dict = {}
    for _ in range(env.max_turns):
        _, reward, terminated, truncated, info = env.step(noop)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "NOOP-only must terminate via asteroid collision"
    assert not won, "loss path must set won=False"
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6, (
        f"cumulative {cumulative} outside [-1, 1]"
    )
