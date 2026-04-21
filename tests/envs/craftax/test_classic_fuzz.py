"""Property-based fuzz tests for Craftax Classic."""

from hypothesis import given, settings
from hypothesis import strategies as st

from glyphbench.envs.craftax.classic import CraftaxClassicEnv


@given(
    actions=st.lists(
        st.integers(min_value=0, max_value=18),
        max_size=500,
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=30, deadline=None)
def test_random_action_sequence_no_crash(actions: list[int], seed: int) -> None:
    env = CraftaxClassicEnv(max_turns=1000)
    obs, info = env.reset(seed=seed)
    assert isinstance(obs, str)
    assert len(obs) > 0
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert isinstance(reward, float)
        assert reward >= 0.0  # can be >1 if multiple achievements unlock
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1
        assert len(grid_lines) == 7
        assert len(grid_lines[0]) == 9
        if terminated or truncated:
            break


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=50, deadline=None)
def test_reset_determinism_across_seeds(seed: int) -> None:
    e1 = CraftaxClassicEnv()
    e2 = CraftaxClassicEnv()
    o1, _ = e1.reset(seed=seed)
    o2, _ = e2.reset(seed=seed)
    assert o1 == o2
