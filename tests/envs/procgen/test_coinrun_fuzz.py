"""Property-based fuzz tests for Procgen CoinRun."""

from hypothesis import given, settings, strategies as st

from atlas_rl.envs.procgen.coinrun import CoinRunEnv


@given(
    actions=st.lists(
        st.integers(min_value=0, max_value=4),
        max_size=200,
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50, deadline=None)
def test_random_action_sequence_no_crash(actions: list[int], seed: int) -> None:
    env = CoinRunEnv(max_turns=512)
    obs, info = env.reset(seed=seed)
    assert isinstance(obs, str)
    assert len(obs) > 0
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert isinstance(reward, float)
        assert reward in (0.0, 10.0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1
        assert len(grid_lines) == 12
        assert len(grid_lines[0]) == 20
        if terminated or truncated:
            break


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=100, deadline=None)
def test_reset_determinism_across_seeds(seed: int) -> None:
    e1 = CoinRunEnv()
    e2 = CoinRunEnv()
    o1, _ = e1.reset(seed=seed)
    o2, _ = e2.reset(seed=seed)
    assert o1 == o2
