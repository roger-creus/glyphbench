import pytest

from rl_world_ascii.envs.dummy.env import DummyEnv


def test_dummy_env_id():
    env = DummyEnv()
    assert env.env_id() == "rl_world_ascii/__dummy-v0"


def test_dummy_env_action_space_has_five_actions():
    env = DummyEnv()
    assert env.action_spec.n == 5
    assert "NORTH" in env.action_spec.names
    assert "NOOP" in env.action_spec.names


def test_dummy_env_reset_places_agent_at_origin():
    env = DummyEnv()
    obs, info = env.reset(seed=0)
    assert "@" in obs
    assert info["turn"] == 0


def test_dummy_env_goal_reachable_in_four_moves():
    env = DummyEnv()
    env.reset(seed=0)
    # two easts, two souths
    east = env.action_spec.index_of("EAST")
    south = env.action_spec.index_of("SOUTH")
    _, r1, t1, _, _ = env.step(east)
    _, r2, t2, _, _ = env.step(east)
    _, r3, t3, _, _ = env.step(south)
    _, r4, t4, _, _ = env.step(south)
    assert not (t1 or t2 or t3)
    assert t4
    assert r4 == 1.0


def test_dummy_env_step_rejects_out_of_range():
    env = DummyEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_dummy_env_truncates_at_max_turns():
    env = DummyEnv(max_turns=3)
    env.reset(seed=0)
    noop = env.action_spec.index_of("NOOP")
    env.step(noop)
    env.step(noop)
    _, _, terminated, truncated, info = env.step(noop)
    assert truncated
    assert not terminated


def test_dummy_env_system_prompt_mentions_goal():
    env = DummyEnv()
    prompt = env.system_prompt()
    assert "goal" in prompt.lower()


def test_dummy_env_rng_determinism():
    e1 = DummyEnv()
    e2 = DummyEnv()
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    assert o1 == o2
