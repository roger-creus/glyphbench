"""Smoke + win/loss path tests for miniatari-yarsrevenge-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-yarsrevenge-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-yarsrevenge-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-yarsrevenge-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Hand-tuned action sequence on seed=0:
#  - tick 1: FIRE the row-5 shield block from the start
#  - ticks 2-7: ladder up to row 3, knock out row-4 then row-3 blocks
#  - ticks 8-15: cross row 5 with the beam clear, kill row-6 and row-7 blocks
#  - ticks 16-23: skim row 7 rightward, then climb to row 5 column 11
#  - tick 24: fire-right kills Qotile (line of sight is open)
_YARSREVENGE_WIN_ACTIONS = [
    "FIRE", "UP", "RIGHT", "FIRE", "UP", "RIGHT", "FIRE",
    "DOWN", "DOWN", "DOWN", "RIGHT", "FIRE",
    "DOWN", "RIGHT", "FIRE",
    "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT",
    "UP", "UP", "RIGHT", "FIRE",
]


def test_yarsrevenge_win_path() -> None:
    """seed=0 + a hand-tuned 24-action sequence breaks all 5 shield blocks
    and shoots Qotile in line of sight from col 11. Asserts terminated,
    won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-yarsrevenge-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _YARSREVENGE_WIN_ACTIONS:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=0"
    assert 0.0 < total <= 1.0 + 1e-9


def test_yarsrevenge_loss_path() -> None:
    """FIRE the row-5 block then NOOP at row 5: the next beam reaches
    the player. Asserts terminated, won=False, cumulative >= -1.0
    (-1 death penalty applies)."""
    env = make_env("glyphbench/miniatari-yarsrevenge-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    _, r, _, _, _ = env.step(ai("FIRE"))
    total += r
    for _ in range(40):
        _, r, terminated, truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path FIRE+NOOP rollout should die to a beam"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
