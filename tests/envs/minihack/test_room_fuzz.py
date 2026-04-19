"""Property-based fuzz tests for MiniHack Room-5x5."""

from hypothesis import given, settings, strategies as st

from atlas_rl.envs.minihack.room import MiniHackRoom5x5Env


@given(
    actions=st.lists(
        st.integers(min_value=0, max_value=14),
        max_size=300,
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50, deadline=None)
def test_random_action_sequence_no_crash(actions: list[int], seed: int) -> None:
    env = MiniHackRoom5x5Env(max_turns=200)
    obs, info = env.reset(seed=seed)
    assert isinstance(obs, str)
    assert len(obs) > 0
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert isinstance(reward, float)
        assert reward >= 0.0
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1
        if terminated or truncated:
            break


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=100, deadline=None)
def test_reset_determinism_across_seeds(seed: int) -> None:
    e1 = MiniHackRoom5x5Env()
    e2 = MiniHackRoom5x5Env()
    o1, _ = e1.reset(seed=seed)
    o2, _ = e2.reset(seed=seed)
    assert o1 == o2
