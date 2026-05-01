"""Smoke + win/loss path tests for miniatari-kangaroo-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-kangaroo-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-kangaroo-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-kangaroo-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_kangaroo_win_path() -> None:
    """Snake-walk up the tower: ladder cols 1, 10, 7 work for floors 0->1, 1->2, 2->3.

    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-kangaroo-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    LEFT = names.index("LEFT")
    RIGHT = names.index("RIGHT")
    UP = names.index("UP")
    # From (2, 12): LEFT to (1, 12), UP*3 -> (1, 9). RIGHT*9 -> (10, 9), UP*3 -> (10, 6).
    # LEFT*3 -> (7, 6), UP*3 -> (7, 3) = floor 3 = win.
    plan = (
        [LEFT]
        + [UP] * 3
        + [RIGHT] * 9
        + [UP] * 3
        + [LEFT] * 3
        + [UP] * 3
    )
    total = 0.0
    won = False
    terminated = False
    for a in plan:
        _obs, r, terminated, _trunc, info = env.step(a)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "win path failed to terminate"
    assert won, f"win path did not win (total={total})"
    assert 0.0 < total <= 1.0 + 1e-6, f"win path total {total} out of (0, 1]"


def test_kangaroo_loss_path() -> None:
    """Climb to floor 1 and stand at col 6 to be hit by the seed=0 coconut.

    On seed=0, coconuts spawn at tick 4k+3 at columns [10, 7, 6, 3, ...].
    The col-6 coconut spawns at tick 11 and reaches row 9 at tick 19.
    Player path: LEFT, UP*3 (-> floor 1), RIGHT*5 (-> col 6), NOOP*N until hit.
    Asserts terminated, won=False, cumulative >= -1.0.
    """
    env = make_env("glyphbench/miniatari-kangaroo-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    LEFT = names.index("LEFT")
    RIGHT = names.index("RIGHT")
    UP = names.index("UP")
    NOOP = names.index("NOOP")
    plan = [LEFT] + [UP] * 3 + [RIGHT] * 5 + [NOOP] * 30
    total = 0.0
    won = True
    terminated = False
    for a in plan:
        _obs, r, terminated, _trunc, info = env.step(a)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "loss path did not terminate"
    assert not won, "loss path unexpectedly won"
    assert total >= -1.0 - 1e-6, f"loss total {total} below -1.0"
    assert total <= 1.0 + 1e-6, f"loss total {total} above 1.0"
