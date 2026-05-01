"""Smoke + win/loss path tests for miniatari-spaceinvaders-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-spaceinvaders-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-spaceinvaders-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-spaceinvaders-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_spaceinvaders_bullet_renders_and_travels() -> None:
    """Regression: FIRE must produce a visible '|' bullet glyph that
    rises one row per tick. Previously FIRE was instantaneous and no
    bullet entity ever appeared."""
    env = make_env("glyphbench/miniatari-spaceinvaders-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    # FIRE -> bullet spawns at cannon row - 1 (y=10), then advances to 9
    # at end of step 1.
    obs, _, _, _, _ = env.step(ai("FIRE"))
    grid_str = obs.grid if hasattr(obs, "grid") else str(obs)
    assert "|" in grid_str, "bullet glyph must render after FIRE"
    # One more NOOP: bullet should have moved up.
    bullet_y_before = env._bullet[1] if env._bullet else None
    obs, _, _, _, _ = env.step(ai("NOOP"))
    bullet_y_after = env._bullet[1] if env._bullet else None
    if bullet_y_before is not None and bullet_y_after is not None:
        assert bullet_y_after < bullet_y_before, (
            "bullet must advance upward each tick"
        )


def test_spaceinvaders_only_one_bullet_in_flight() -> None:
    """Regression: a second FIRE while a bullet is already in flight
    should be a no-op (the player has only one bullet)."""
    env = make_env("glyphbench/miniatari-spaceinvaders-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    env.step(ai("FIRE"))
    bullet1 = env._bullet
    assert bullet1 is not None
    env.step(ai("FIRE"))  # should be ignored
    bullet2 = env._bullet
    # Bullet still in flight, just advanced by one row from the
    # original spawn point.
    assert bullet2 is not None
    assert bullet2[0] == bullet1[0]


# Captured by replaying a column-tracker policy on seed=0 against the
# slowed-down wave (advance_every=24, sweep_every=5). The agent FIREs
# while standing in an invader column, then pre-positions to the next
# column while the bullet is in flight. Wins 8 kills in 143 ticks.
_SI_WIN_ACTIONS_SEED0 = (
    "FIRE LEFT LEFT NOOP NOOP LEFT NOOP NOOP NOOP NOOP LEFT FIRE RIGHT "
    "RIGHT NOOP LEFT NOOP NOOP NOOP NOOP RIGHT NOOP FIRE RIGHT RIGHT LEFT "
    "NOOP NOOP NOOP NOOP RIGHT NOOP NOOP FIRE RIGHT RIGHT NOOP NOOP NOOP "
    "NOOP LEFT FIRE LEFT LEFT NOOP LEFT NOOP NOOP NOOP NOOP LEFT NOOP "
    "FIRE RIGHT RIGHT LEFT NOOP NOOP NOOP NOOP RIGHT NOOP NOOP FIRE LEFT "
    "NOOP NOOP NOOP NOOP NOOP LEFT NOOP NOOP NOOP FIRE LEFT LEFT NOOP "
    "FIRE LEFT NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT NOOP FIRE LEFT "
    "NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT NOOP NOOP LEFT FIRE RIGHT "
    "RIGHT NOOP RIGHT FIRE RIGHT RIGHT FIRE LEFT NOOP NOOP NOOP NOOP "
    "NOOP NOOP NOOP NOOP NOOP LEFT FIRE RIGHT RIGHT FIRE LEFT NOOP NOOP "
    "NOOP NOOP LEFT NOOP NOOP NOOP NOOP LEFT FIRE RIGHT RIGHT FIRE LEFT "
    "FIRE NOOP"
).split()


def test_spaceinvaders_win_path() -> None:
    """Drive a precomputed column-tracker action sequence on seed=0 to
    clear all 8 invaders. Asserts terminated, won=True, cumulative == 1.0."""
    env = make_env("glyphbench/miniatari-spaceinvaders-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _SI_WIN_ACTIONS_SEED0:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9
    assert abs(total - 1.0) < 1e-9


def test_spaceinvaders_loss_path() -> None:
    """NOOP-only: the wave reaches the cannon row at ~tick 144 with no
    kills, terminating with a -1. Asserts terminated, won=False,
    cumulative >= -1.0 (the death penalty preserves any kill bonus)."""
    env = make_env("glyphbench/miniatari-spaceinvaders-v0")
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
    assert terminated, "NOOP rollout should terminate via wave breach"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
