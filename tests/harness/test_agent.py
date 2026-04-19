import pytest

from atlas_rl.envs.dummy.env import DummyEnv
from atlas_rl.harness.agent import HarnessAgent
from atlas_rl.harness.mock_client import MockLLMClient, ScriptedResponse


@pytest.mark.asyncio
async def test_agent_completes_dummy_env_episode_with_scripted_llm():
    # Scripted policy: go east, east, south, south — should hit the goal
    responses = [
        ScriptedResponse(text='{"action": "EAST", "thinking": "go right"}'),
        ScriptedResponse(text='{"action": "EAST"}'),
        ScriptedResponse(text='{"action": "SOUTH"}'),
        ScriptedResponse(text='{"action": "SOUTH"}'),
    ]
    client = MockLLMClient(scripted=responses)
    env = DummyEnv()
    agent = HarnessAgent(
        env=env,
        client=client,
        temperature=0.0,
        max_output_tokens=512,
    )
    episode_return, episode_length, turn_metrics = await agent.run_episode(seed=0)
    assert episode_return == 1.0
    assert episode_length == 4
    assert len(turn_metrics) == 4
    # No parse errors
    assert all(not m.action_parse_error for m in turn_metrics)


@pytest.mark.asyncio
async def test_agent_retries_on_malformed_then_falls_back_to_noop():
    # Three malformed responses, then the fourth would have been valid — but
    # MAX_REPAIR_RETRIES is 3 so we fall back to NOOP.
    responses = [
        ScriptedResponse(text="not json"),
        ScriptedResponse(text="still not json"),
        ScriptedResponse(text="still not json 2"),
        ScriptedResponse(text="still not json 3"),  # this 4th attempt exists; 4 total
    ]
    client = MockLLMClient(scripted=responses)
    env = DummyEnv(max_turns=1)
    agent = HarnessAgent(env=env, client=client, temperature=0.0, max_output_tokens=512)
    _, _, turn_metrics = await agent.run_episode(seed=0)
    assert len(turn_metrics) == 1
    m = turn_metrics[0]
    assert m.action_parse_error is True
    assert m.action_fell_back_to_noop is True
    assert m.action_parse_retries == 3  # exactly 3 retries after the first attempt


@pytest.mark.asyncio
async def test_agent_updates_persistent_state_across_turns():
    responses = [
        ScriptedResponse(text='{"action": "EAST", "strategic_plan_update": "go east twice", "subgoals_update": {"add": ["reach goal"]}, "lessons_to_add": ["east is right"]}'),
        ScriptedResponse(text='{"action": "EAST", "subgoals_update": {"mark_done": [0]}}'),
        ScriptedResponse(text='{"action": "SOUTH"}'),
        ScriptedResponse(text='{"action": "SOUTH"}'),
    ]
    client = MockLLMClient(scripted=responses)
    env = DummyEnv()
    agent = HarnessAgent(env=env, client=client, temperature=0.0, max_output_tokens=512)
    await agent.run_episode(seed=0)
    # After episode, the agent's state should reflect the updates
    assert agent.state.strategic_plan == "go east twice"
    assert "east is right" in agent.state.lessons
    assert any(sg.done for sg in agent.state.subgoals)
