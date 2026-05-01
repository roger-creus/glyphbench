"""Smoke + win/loss path tests for miniatari-battlezone-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-battlezone-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-battlezone-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-battlezone-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Scripted win path tracked down on seed=16 (where the three corner
# enemies happen to be (12,1), (1,1), (12,10)). On seed=0 the corner
# layout combined with the every-6-tick shell barrage makes a clean
# scripted clear hard; this seed-locked sequence reliably destroys all
# 3 enemy tanks in 21 steps.
_BATTLEZONE_WIN_SEQUENCE_SEED16 = [
    "DOWN", "DOWN", "DOWN", "DOWN", "RIGHT", "FIRE", "UP", "RIGHT",
    "UP", "RIGHT", "UP", "UP", "RIGHT", "UP", "FIRE", "LEFT", "LEFT",
    "UP", "UP", "LEFT", "FIRE",
]


def test_battlezone_win_path() -> None:
    """Scripted ride-and-fire on seed=16 destroys all 3 enemy tanks."""
    env = make_env("glyphbench/miniatari-battlezone-v0")
    env.reset(seed=16)
    cumulative = 0.0
    won = False
    terminated = False
    info: dict = {}
    for name in _BATTLEZONE_WIN_SEQUENCE_SEED16:
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


def test_battlezone_loss_path() -> None:
    """NOOP-only lets the enemies close in => shell or ram death (-1)."""
    env = make_env("glyphbench/miniatari-battlezone-v0")
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
    assert terminated, "NOOP-only must terminate via shell or ram"
    assert not won, "loss path must set won=False"
    assert -1.0 - 1e-6 <= cumulative, f"cumulative {cumulative} below -1"
