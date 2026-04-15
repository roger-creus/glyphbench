import pytest

from rl_world_ascii.envs.dummy.env import DummyEnv
from rl_world_ascii.runner.random_agent import RandomAgent


@pytest.mark.asyncio
async def test_random_agent_runs_episode_and_returns_metrics():
    env = DummyEnv(max_turns=20)
    agent = RandomAgent(env=env, seed=0)
    episode_return, length, turn_metrics = await agent.run_episode(seed=0)
    assert length >= 1
    assert len(turn_metrics) == length
    assert all(m.dollar_cost_turn == 0.0 for m in turn_metrics)
    assert all(m.tokens_in == 0 for m in turn_metrics)


@pytest.mark.asyncio
async def test_random_agent_seed_determinism():
    env1 = DummyEnv(max_turns=20)
    env2 = DummyEnv(max_turns=20)
    a1 = RandomAgent(env=env1, seed=7)
    a2 = RandomAgent(env=env2, seed=7)
    r1, l1, m1 = await a1.run_episode(seed=0)
    r2, l2, m2 = await a2.run_episode(seed=0)
    assert l1 == l2
    assert [m.action_name for m in m1] == [m.action_name for m in m2]
