"""Smoke + win/loss path tests for miniatari-bankheist-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-bankheist-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-bankheist-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-bankheist-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Scripted bank route on seed=0: rob (11,6) -> drive across to (4,1)
# while keeping clear of the two cops in opposite corners -> finish at
# (1,1). 20 steps, cum=+1.0.
_BANKHEIST_WIN_SEQUENCE_SEED0 = [
    "RIGHT", "RIGHT", "RIGHT", "RIGHT",
    "DOWN", "LEFT", "LEFT", "LEFT",
    "UP", "LEFT", "UP", "LEFT", "UP", "LEFT",
    "UP", "LEFT", "UP", "LEFT", "LEFT", "LEFT",
]


def test_bankheist_win_path() -> None:
    """Scripted route on seed=0 robs all 3 banks while dodging both cops."""
    env = make_env("glyphbench/miniatari-bankheist-v0")
    env.reset(seed=0)
    cumulative = 0.0
    won = False
    terminated = False
    info: dict = {}
    for name in _BANKHEIST_WIN_SEQUENCE_SEED0:
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


def test_bankheist_loss_path() -> None:
    """NOOP-only lets the cops converge => -1 terminal penalty."""
    env = make_env("glyphbench/miniatari-bankheist-v0")
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
    assert terminated, "NOOP-only must terminate via cop catch"
    assert not won, "loss path must set won=False"
    assert -1.0 - 1e-6 <= cumulative, f"cumulative {cumulative} below -1"
