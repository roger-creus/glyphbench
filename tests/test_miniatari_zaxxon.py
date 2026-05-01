"""Smoke + win/loss path tests for miniatari-zaxxon-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-zaxxon-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-zaxxon-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-zaxxon-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Action sequence captured from a row-aligned FIRE strategy on seed=2:
# weave UP/DOWN to share each turret's row, then FIRE.
_ZAXXON_WIN_ACTIONS = [
    "FIRE", "DOWN", "FIRE", "DOWN", "FIRE",
    "UP", "UP", "UP", "UP",
    "FIRE", "FIRE",
    "UP", "UP",
    "FIRE",
]


def test_zaxxon_win_path() -> None:
    """seed=2 + a 14-action sequence kills all 6 turrets while never
    sharing a row with a live turret bullet. Asserts terminated,
    won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-zaxxon-v0")
    env.reset(seed=2)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _ZAXXON_WIN_ACTIONS:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=2"
    assert 0.0 < total <= 1.0 + 1e-9


def test_zaxxon_loss_path() -> None:
    """Move DOWN to row 6 (turret at (13,6) on seed=0) then NOOP: the
    next bullet from that turret reaches and kills the player.
    Asserts terminated, won=False, cumulative >= -1.0."""
    env = make_env("glyphbench/miniatari-zaxxon-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    _, r, _, _, _ = env.step(ai("DOWN"))
    total += r
    for _ in range(50):
        _, r, terminated, truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path DOWN+NOOP rollout should be hit by a bullet"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
