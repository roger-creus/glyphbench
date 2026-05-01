"""Smoke + win/loss path tests for miniatari-pong-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-pong-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-pong-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-pong-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_pong_paddle_render_matches_collision() -> None:
    """Regression: paddle render and collision both span 3 cells.

    Previously the paddle rendered 4 cells (range(-2, 2) due to floor
    division of -3//2) but the collision check spanned only 3. That
    let the ball pass through the visible bottom paddle cell.
    """
    env = make_env("glyphbench/miniatari-pong-v0")
    env.reset(seed=0)
    grid = env._render_current_observation().grid
    rows = grid.split("\n")
    y_count = sum(1 for row in rows if "Y" in row)
    p_count = sum(1 for row in rows if "P" in row)
    assert y_count == 3, f"expected 3-cell agent paddle, got {y_count}"
    assert p_count == 3, f"expected 3-cell opponent paddle, got {p_count}"


# Captured by replaying a ball-tracker policy on seed=14: the agent paddle
# moves to follow the ball each tick. With the post-fix 3-cell paddle the
# tracker wins 3-0 in 43 steps. Sequence is deterministic given the
# scripted opponent's same-seed RNG.
_PONG_WIN_ACTIONS_SEED14 = (
    "NOOP UP UP UP NOOP DOWN DOWN NOOP DOWN DOWN DOWN DOWN NOOP UP UP UP UP "
    "UP UP NOOP DOWN DOWN DOWN DOWN DOWN UP UP DOWN DOWN DOWN NOOP UP UP UP "
    "UP UP UP NOOP DOWN DOWN DOWN DOWN DOWN"
).split()


def test_pong_win_path() -> None:
    """Drive a ball-tracker action sequence on seed=14 to win 3-0.
    Asserts terminated, won=True, 0 < cumulative <= 1.0."""
    env = make_env("glyphbench/miniatari-pong-v0")
    env.reset(seed=14)
    ai = env.action_spec.names.index
    total = 0.0
    terminated = False
    won = False
    for name in _PONG_WIN_ACTIONS_SEED14:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True on seed=14"
    assert 0.0 < total <= 1.0 + 1e-9


def test_pong_loss_path() -> None:
    """NOOP-only on seed=3 gives the opponent enough wins to terminate
    with a loss (the agent paddle never moves and the opponent scores 3).
    Asserts terminated, won=False, cumulative >= -1.0."""
    env = make_env("glyphbench/miniatari-pong-v0")
    env.reset(seed=3)
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
    assert terminated, "loss-path NOOP rollout should terminate"
    assert not won, "loss-path should set won=False"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
