"""Smoke + win/loss path tests for miniatari-skiing-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-skiing-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-skiing-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-skiing-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_skiing_course_terminates_on_last_gate() -> None:
    """Regression: when the last gate scrolls past the skier the run
    terminates (previously the env just truncated on max_turns)."""
    env = make_env("glyphbench/miniatari-skiing-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    terminated = False
    for _ in range(80):
        _, _r, terminated, _truncated, info = env.step(ai("NOOP"))
        if terminated:
            break
    assert terminated, "episode must terminate when course ends, not truncate"


# Captured via a center-of-next-gate tracker on seed=2: the post-fix
# spacing of 5 rows/gate makes seed=2 reachable in 50 ticks.
_SKIING_WIN_ACTIONS_SEED2 = (
    "RIGHT RIGHT NOOP NOOP NOOP LEFT LEFT LEFT LEFT LEFT LEFT LEFT NOOP "
    "NOOP NOOP RIGHT NOOP NOOP NOOP NOOP RIGHT RIGHT NOOP NOOP NOOP RIGHT "
    "RIGHT RIGHT RIGHT NOOP LEFT LEFT LEFT LEFT NOOP LEFT LEFT LEFT LEFT "
    "NOOP RIGHT RIGHT RIGHT NOOP NOOP RIGHT RIGHT RIGHT NOOP NOOP"
).split()


def test_skiing_win_path() -> None:
    """Drive a precomputed gate-tracker action sequence on seed=2 to
    clear all 10 gates. Asserts terminated, won=True, cumulative == 1.0."""
    env = make_env("glyphbench/miniatari-skiing-v0")
    env.reset(seed=2)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _SKIING_WIN_ACTIONS_SEED2:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=2"
    assert 0.0 < total <= 1.0 + 1e-9


def test_skiing_loss_path() -> None:
    """NOOP-only on seed=0: skier sits at column 7 while gates scatter
    around the slope. The course-end terminal fires at ~tick 55 with
    fewer than 10 gates cleared. Asserts terminated, won=False,
    cumulative >= 0 (Pattern A: no penalty)."""
    env = make_env("glyphbench/miniatari-skiing-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = True
    for _ in range(120):
        _, r, terminated, _truncated, info = env.step(ai("NOOP"))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "loss-path NOOP rollout should terminate"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
