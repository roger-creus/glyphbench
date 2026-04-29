"""Tests for GlyphbenchMultiTurnEnv + load_environment."""

from __future__ import annotations

import json
from typing import Any

import pytest
import verifiers as vf
from verifiers.types import Response, ResponseMessage, ResponseTokens

from glyphbench.verifiers_integration import GlyphbenchMultiTurnEnv, load_environment


class _NoopClient(vf.Client):
    """Client placeholder; tests monkeypatch env.get_model_response directly."""

    def setup_client(self, config: Any) -> Any:
        return config

    async def to_native_tool(self, tool: vf.Tool) -> vf.Tool:
        return tool

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(
        self,
        prompt,
        model: str,
        sampling_args: dict,
        tools: list | None = None,
        **kwargs,
    ):
        raise AssertionError("get_native_response should not be called")

    async def raise_from_native_response(self, response) -> None:
        return None

    async def from_native_response(self, response) -> Response:
        return response

    async def close(self) -> None:
        return None


def _response(
    content: str,
    *,
    reasoning_content: str | None = None,
    prompt_ids: list[int] | None = None,
    completion_ids: list[int] | None = None,
    is_truncated: bool = False,
) -> Response:
    tokens = None
    if prompt_ids is not None and completion_ids is not None:
        tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[0] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=[-0.1 * i for i, _ in enumerate(completion_ids, 1)],
        )
    return Response(
        id="test",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=None,
            finish_reason="length" if is_truncated else "stop",
            is_truncated=is_truncated,
            tokens=tokens,
        ),
    )


def test_load_environment_returns_multi_turn_env():
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=2,
        n_frames=4,
        max_output_tokens=512,
    )
    assert isinstance(env, GlyphbenchMultiTurnEnv)


def test_load_environment_memory_args_are_stored():
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        use_memory=True,
        memory_update_max_tokens=128,
    )
    assert isinstance(env, GlyphbenchMultiTurnEnv)
    assert env._use_memory is True
    assert env._memory_update_max_tokens == 128


def test_load_environment_dataset_shape():
    env = load_environment(task_id="glyphbench/__dummy-v0", num_episodes=3)
    assert len(env.dataset) == 3
    for row in env.dataset:
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        assert info["env_id"] == "glyphbench/__dummy-v0"
        assert isinstance(info["seed"], int)
    # Seeds should be distinct.
    seeds = [
        json.loads(r["info"])["seed"] if isinstance(r["info"], str) else r["info"]["seed"]
        for r in env.dataset
    ]
    assert len(set(seeds)) == len(seeds)


def test_load_environment_multiple_env_ids():
    env = load_environment(
        task_id=["glyphbench/__dummy-v0"],
        num_episodes=2,
    )
    assert len(env.dataset) == 2


def test_load_environment_rejects_unknown_id():
    with pytest.raises(KeyError):
        load_environment(task_id="glyphbench/does-not-exist-v0", num_episodes=1)


@pytest.mark.asyncio
async def test_setup_state_creates_game_and_initial_obs():
    env = load_environment(task_id="glyphbench/__dummy-v0", num_episodes=1)
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    assert "game" in state
    assert state["current_obs"]  # non-empty
    assert state["parse_failures"] == 0
    assert state["episode_return"] == 0.0
    assert state["done"] is False
    assert state["terminated"] is False
    assert state["truncated"] is False
    # The prompt must have been populated with system + initial user turn.
    assert len(state["prompt"]) == 2
    assert state["prompt"][0]["role"] == "system"
    assert state["prompt"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_setup_state_includes_empty_memory_block_when_enabled():
    env = load_environment(
        task_id="glyphbench/__dummy-v0", num_episodes=1, use_memory=True
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    assert state["memory_enabled"] is True
    assert state["memory"] == ""
    assert "[Memory]" in state["prompt"][1]["content"]
    assert "<memory>\n\n</memory>" in state["prompt"][1]["content"]


@pytest.mark.asyncio
async def test_env_response_applies_action_and_updates_state():
    # Pass n_frames=4 explicitly so the test exercises the frame-stacking
    # accumulator. The harness default is n_frames=0 (stateless per turn),
    # which would silently drop appended frames via deque(maxlen=0).
    env = load_environment(
        task_id="glyphbench/__dummy-v0", num_episodes=1, n_frames=4
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [{"reward": None, "extras": {}}],  # verifiers appends before env_response
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    state["trajectory"] = [{"reward": None, "extras": {}}]
    model_reply = "<think>go east</think><action>EAST</action>"
    messages = [{"role": "assistant", "content": model_reply}]
    response = await env.env_response(messages, state)
    # Returns a list of 1 user message with the next observation.
    assert isinstance(response, list) and len(response) == 1
    assert response[0]["role"] == "user"
    assert state["trajectory"][-1]["reward"] is not None
    # One frame accumulated for the action just taken.
    assert len(state["frames"]) == 1


@pytest.mark.asyncio
async def test_memory_add_model_response_combines_action_and_memory_turn(monkeypatch):
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        use_memory=True,
        memory_update_max_tokens=3,
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
        "sampling_args": {"max_tokens": 9},
    }
    state = await env.setup_state(state)
    prompt_messages = state["prompt"]

    memory_response = _response(
        "<think>revise</think><memory>moved east once</memory>",
        prompt_ids=[1, 2, 3, 4, 5, 6],
        completion_ids=[7, 8],
    )
    seen: dict = {}

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        seen["state"] = state_arg
        seen["prompt"] = prompt_arg
        seen["sampling_args"] = kwargs.get("sampling_args")
        return memory_response

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)
    action_response = _response(
        "<think>go east</think><action>EAST</action>",
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
    )

    await env.add_model_response(state, prompt_messages, action_response)

    assert len(state["trajectory"]) == 1
    assert state["memory"] == "moved east once"
    assert seen["sampling_args"] == {
        "max_tokens": 3,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    assert seen["prompt"][-1]["role"] == "user"
    memory_update_prompt = seen["prompt"][-1]["content"]
    assert "Parsed action: EAST" in memory_update_prompt
    assert "<think>go east</think><action>EAST</action>" in memory_update_prompt
    assert "[Next Observation]" in memory_update_prompt
    assert "[Grid]" in memory_update_prompt
    assert "[HUD]" not in memory_update_prompt
    assert "Pos:" not in memory_update_prompt

    step = state["trajectory"][0]
    assert [m["role"] for m in step["completion"]] == ["assistant", "user", "assistant"]
    assert step["reward"] == 0.0
    assert step["extras"]["glyphbench_memory"]["previous_memory"] == ""
    assert step["extras"]["glyphbench_memory"]["stored_memory"] == "moved east once"
    assert step["extras"]["glyphbench_memory"]["extraction_mode"] == "tag"
    assert step["tokens"]["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert step["tokens"]["completion_mask"] == [1, 1, 0, 0, 1, 1]

    next_prompt = await env.get_prompt_messages(state)
    assert len(next_prompt) == 2
    assert "<memory>\nmoved east once\n</memory>" in next_prompt[1]["content"]


@pytest.mark.asyncio
async def test_memory_add_model_response_keeps_step_without_token_data(monkeypatch):
    env = load_environment(
        task_id="glyphbench/__dummy-v0", num_episodes=1, use_memory=True
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
        "sampling_args": {},
    }
    state = await env.setup_state(state)

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        return _response("<think>update</think>wall north")

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)
    await env.add_model_response(
        state,
        state["prompt"],
        _response("<action>NORTH</action>"),
    )

    assert len(state["trajectory"]) == 1
    assert state["trajectory"][0]["tokens"] is None
    assert state["memory"] == "wall north"


@pytest.mark.asyncio
async def test_memory_update_prompt_includes_separate_action_reasoning(monkeypatch):
    env = load_environment(
        task_id="glyphbench/__dummy-v0", num_episodes=1, use_memory=True
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
        "sampling_args": {},
    }
    state = await env.setup_state(state)
    seen: dict = {}

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        seen["prompt"] = prompt_arg
        return _response("<memory>remembered</memory>")

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)
    await env.add_model_response(
        state,
        state["prompt"],
        _response(
            "<action>EAST</action>",
            reasoning_content="The goal is east, so move east.",
        ),
    )

    memory_update_prompt = seen["prompt"][-1]["content"]
    assert "<think>\nThe goal is east, so move east.\n</think>" in memory_update_prompt
    assert "<action>EAST</action>" in memory_update_prompt


@pytest.mark.asyncio
async def test_memory_rollout_uses_two_model_calls_for_one_env_turn(monkeypatch):
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        max_turns=1,
        use_memory=True,
    )
    row = env.dataset[0]
    calls: list[list[dict]] = []

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        calls.append(prompt_arg)
        if len(calls) == 1:
            assert state_arg["trajectory"] == []
            return _response("<think>try east</think><action>EAST</action>")
        assert len(calls) == 2
        assert len(state_arg["trajectory"]) == 1
        assert [m["role"] for m in state_arg["trajectory"][0]["completion"]] == [
            "assistant"
        ]
        assert prompt_arg[-1]["role"] == "user"
        return _response("<think>update</think><memory>moved east</memory>")

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)

    state = await env.rollout(
        input=row,
        client=_NoopClient(object()),
        model="test-model",
        sampling_args={"max_tokens": 16},
    )

    assert len(calls) == 2
    assert len(state["trajectory"]) == 1
    assert state["memory"] == "moved east"
    assert state["stop_condition"] == "has_final_env_response"
    assert [m["role"] for m in state["trajectory"][0]["completion"]] == [
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_is_done_terminates_on_game_end():
    env = load_environment(task_id="glyphbench/__dummy-v0", num_episodes=1)
    state: dict = {"done": False}
    assert await env.is_done(state) is False
    state["done"] = True
    assert await env.is_done(state) is True
