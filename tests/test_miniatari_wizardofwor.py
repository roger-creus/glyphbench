"""Smoke + win/loss path tests for miniatari-wizardofwor-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-wizardofwor-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-wizardofwor-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-wizardofwor-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Action sequence captured from a starting-cell camp + line-of-sight FIRE
# strategy on seed=1: the agent never moves and shoots monsters as they
# walk into the firing column.
_WIZARDOFWOR_WIN_ACTIONS = [
    "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
    "FIRE",
    "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
    "FIRE",
    "NOOP", "NOOP",
    "FIRE",
    "NOOP", "NOOP",
    "FIRE",
    "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
    "FIRE",
]


def test_wizardofwor_win_path() -> None:
    """seed=1 + a NOOP/FIRE sequence kills all 5 monsters from the
    starting cell. Asserts terminated, won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-wizardofwor-v0")
    env.reset(seed=1)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _WIZARDOFWOR_WIN_ACTIONS:
        _, r, terminated, truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=1"
    assert 0.0 < total <= 1.0 + 1e-9


def test_wizardofwor_loss_path() -> None:
    """NOOP only on seed=0: a monster reaches the player and triggers the
    -1 death reward. Asserts terminated, won=False, cumulative >= -1.0."""
    env = make_env("glyphbench/miniatari-wizardofwor-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(60):
        _, r, terminated, truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path NOOP rollout should terminate via catch"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
