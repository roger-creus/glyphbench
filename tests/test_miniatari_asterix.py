"""Smoke + win/loss path tests for miniatari-asterix-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-asterix-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-asterix-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-asterix-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Scripted enemy-avoiding helmet collector for seed=2.
# Asterix's seed=0 starts with enemies adjacent enough that every
# straight-line path costs catches; seed=2 places enemies far enough
# from the player that this pinned sequence cleanly collects all 5
# helmets in 42 steps.
_ASTERIX_WIN_SEQUENCE_SEED2 = [
    "LEFT", "LEFT", "DOWN", "DOWN", "UP", "UP", "UP", "UP", "UP",
    "RIGHT", "UP", "LEFT", "LEFT", "LEFT", "DOWN", "LEFT", "DOWN",
    "RIGHT", "RIGHT", "LEFT", "RIGHT", "LEFT", "LEFT", "RIGHT",
    "LEFT", "LEFT", "RIGHT", "LEFT", "UP", "DOWN", "DOWN", "RIGHT",
    "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT",
    "RIGHT", "RIGHT", "UP",
]


def test_asterix_win_path() -> None:
    """Scripted dodge-and-collect on seed=2 picks up all 5 helmets."""
    env = make_env("glyphbench/miniatari-asterix-v0")
    env.reset(seed=2)
    cumulative = 0.0
    won = False
    terminated = False
    info: dict = {}
    for name in _ASTERIX_WIN_SEQUENCE_SEED2:
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


def test_asterix_loss_path() -> None:
    """NOOP-only lets enemies converge and catch the player => -1 terminal."""
    env = make_env("glyphbench/miniatari-asterix-v0")
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
    assert terminated, "NOOP-only must terminate via enemy catch"
    assert not won, "loss path must set won=False"
    assert -1.0 - 1e-6 <= cumulative, f"cumulative {cumulative} below -1"
