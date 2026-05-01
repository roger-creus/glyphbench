"""Property-based fuzz tests for Atari Pong."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from glyphbench.envs.atari.pong import PongEnv


@given(
    actions=st.lists(
        st.integers(min_value=0, max_value=5),
        max_size=300,
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50, deadline=None)
def test_random_action_sequence_no_crash(actions: list[int], seed: int) -> None:
    env = PongEnv(max_turns=500)
    obs, info = env.reset(seed)
    assert isinstance(obs, str)
    assert len(obs) > 0
    # Pattern C reward design (post-2026-05-01): ±1/_WIN_TARGET per point,
    # 0 otherwise. Permit a small float tolerance.
    per_point = 1.0 / env._WIN_TARGET
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert isinstance(reward, float)
        assert (
            math.isclose(reward, 0.0, abs_tol=1e-9)
            or math.isclose(reward, per_point, abs_tol=1e-9)
            or math.isclose(reward, -per_point, abs_tol=1e-9)
        ), f"unexpected reward magnitude {reward}"
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1
        if terminated or truncated:
            break


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=100, deadline=None)
def test_reset_determinism_across_seeds(seed: int) -> None:
    e1 = PongEnv()
    e2 = PongEnv()
    o1, _ = e1.reset(seed)
    o2, _ = e2.reset(seed)
    assert o1 == o2
