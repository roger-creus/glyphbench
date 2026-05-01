"""Smoke + win/loss path tests for miniatari-amidar-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-amidar-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-amidar-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-amidar-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Walk the perimeter clockwise from start (3,2). On seed=0 the patrol
# is at (5,6) step 5 of 10; the path below paints all 10 cells while
# letting the patrol pass before re-entering its range.
# Sequence from (3,2): D (3,3) R (4,3) R (5,3) D (5,4) D (5,5) D (5,6)
#                     LEFT to wait, then thread back around.
_AMIDAR_WIN_SEQUENCE_SEED0 = [
    "DOWN", "RIGHT", "RIGHT",          # paint top edge (3 cells)
    "DOWN", "DOWN", "DOWN",            # paint right edge (3 cells)
    "LEFT",                            # paint (4,6) — 7 painted
    "NOOP", "NOOP",                    # let patrol move past (3,6)
    "LEFT",                            # paint (3,6) — 8 painted
    "NOOP", "NOOP",                    # let patrol move to (3,4)
    "UP",                              # paint (3,5) — 9 painted
    "NOOP", "NOOP",                    # let patrol move to (3,3)
    "UP",                              # paint (3,4) — 10 painted -> WIN
]


def test_amidar_win_path() -> None:
    """Scripted clockwise paint with patrol-avoiding pauses wins on seed=0."""
    env = make_env("glyphbench/miniatari-amidar-v0")
    env.reset(seed=0)
    cumulative = 0.0
    won = False
    terminated = False
    info: dict = {}
    for name in _AMIDAR_WIN_SEQUENCE_SEED0:
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


# Walk straight onto the patrol's cell.
_AMIDAR_LOSS_SEQUENCE_SEED0 = ["DOWN", "DOWN", "DOWN", "DOWN", "RIGHT"]


def test_amidar_loss_path() -> None:
    """Charging into the patrol's cell triggers the -1 terminal penalty."""
    env = make_env("glyphbench/miniatari-amidar-v0")
    env.reset(seed=0)
    cumulative = 0.0
    won = True
    terminated = False
    info: dict = {}
    for name in _AMIDAR_LOSS_SEQUENCE_SEED0:
        action = env.action_spec.names.index(name)
        _, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss sequence must terminate (caught by patrol)"
    assert not won, "loss path must set won=False"
    assert -1.0 - 1e-6 <= cumulative, f"cumulative {cumulative} below -1"
