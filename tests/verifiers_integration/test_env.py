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


def test_load_environment_dataset_task_matches_env_id():
    env = load_environment(task_id="glyphbench/__dummy-v0", num_episodes=2)
    for row in env.dataset:
        assert row["task"] == "glyphbench/__dummy-v0"
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        assert row["task"] == info["env_id"]

    env_multi = load_environment(
        task_id=["glyphbench/__dummy-v0"], num_episodes=2
    )
    for row in env_multi.dataset:
        info = (
            json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        )
        assert row["task"] == info["env_id"]


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
    await env.setup_state(state)
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
    await env.setup_state(state)
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
    await env.setup_state(state)
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
async def test_memory_add_model_response_emits_two_steps(monkeypatch):
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
    await env.setup_state(state)
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

    # One env turn → two trajectory steps (action + memory). Each step's
    # completion is purely assistant tokens, which prime-rl's pretokenize
    # path requires.
    assert len(state["trajectory"]) == 2
    assert state["num_turns"] == 1
    assert state["memory"] == "moved east once"
    assert seen["sampling_args"] == {
        "max_tokens": 3,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    assert seen["prompt"][-1]["role"] == "user"
    memory_update_prompt = seen["prompt"][-1]["content"]
    # Memory_user has all four sections.
    assert "[Last Action]" in memory_update_prompt
    assert "[Env Response]" in memory_update_prompt
    assert "[Next Observation]" in memory_update_prompt
    assert "[Memory Update]" in memory_update_prompt
    assert "Reward: +0.000" in memory_update_prompt
    # Action transcript is re-injected (workaround for chat-template <think>
    # stripping) so the memory writer can use the action turn's reasoning.
    assert "<action>EAST</action>" in memory_update_prompt
    assert "Action applied: EAST" in memory_update_prompt
    assert "Parse status: ok" in memory_update_prompt
    assert "Output truncated: false" in memory_update_prompt
    # Next obs grid surfaces in the [Next Observation] block.
    assert "[Grid]" in memory_update_prompt
    # Removed legacy block names should NOT appear (these were the pre-rework
    # injection points; the new format uses [Last Action] / [Env Response]).
    assert "[Previous Memory]" not in memory_update_prompt
    assert "[Action Response]" not in memory_update_prompt
    assert "Parsed action:" not in memory_update_prompt

    action_step, memory_step = state["trajectory"]
    assert [m["role"] for m in action_step["completion"]] == ["assistant"]
    assert action_step["reward"] == 0.0
    assert action_step["extras"]["glyphbench_step_role"] == "action"
    assert action_step["extras"]["parse_failed"] is False
    assert action_step["extras"]["action_chosen"] == "EAST"
    assert action_step["extras"]["forfeit"] is False
    assert action_step["tokens"]["completion_ids"] == [3, 4]

    # memory_step's prompt extends the action_step's prompt+completion with
    # the new memory_update_user message, and its completion is the
    # assistant memory response only.
    assert memory_step["prompt"][: len(prompt_messages)] == list(prompt_messages)
    assert memory_step["prompt"][-1]["role"] == "user"
    assert "[Memory Update]" in memory_step["prompt"][-1]["content"]
    assert "[Last Action]" in memory_step["prompt"][-1]["content"]
    assert "[Next Observation]" in memory_step["prompt"][-1]["content"]
    assert [m["role"] for m in memory_step["completion"]] == ["assistant"]
    assert memory_step["reward"] == 0.0  # same per-turn reward
    assert memory_step["extras"]["glyphbench_step_role"] == "memory"
    assert memory_step["extras"]["memory_parse_failed"] is False
    assert memory_step["extras"]["stored_memory"] == "moved east once"
    assert memory_step["tokens"]["completion_ids"] == [7, 8]

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
    await env.setup_state(state)

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        return _response("<memory>wall north</memory>")

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)
    await env.add_model_response(
        state,
        state["prompt"],
        _response("<action>NORTH</action>"),
    )

    # Both action_step and memory_step exist; both lack token data because
    # the fake responses don't carry any.
    assert len(state["trajectory"]) == 2
    assert state["trajectory"][0]["tokens"] is None
    assert state["trajectory"][1]["tokens"] is None
    assert state["memory"] == "wall north"


@pytest.mark.asyncio
async def test_memory_update_prompt_reinjects_action_reasoning(monkeypatch):
    """Action turn's reasoning + tag is re-injected as plain text in the
    memory_user. The Qwen3.5 chat template strips ``<think>`` from prior
    assistant turns by default, so this re-injection is the only way the
    memory writer sees the reasoning."""
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
    await env.setup_state(state)
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
    # All four sections present.
    assert "[Last Action]" in memory_update_prompt
    assert "[Env Response]" in memory_update_prompt
    assert "[Next Observation]" in memory_update_prompt
    assert "[Memory Update]" in memory_update_prompt
    # Reasoning + tag re-injected as plain text in [Last Action] block.
    assert "<think>" in memory_update_prompt
    assert "The goal is east" in memory_update_prompt
    assert "<action>EAST</action>" in memory_update_prompt
    assert "Action applied: EAST" in memory_update_prompt
    # Old block names are gone (replaced by [Last Action] / [Env Response]).
    assert "[Action Response]" not in memory_update_prompt
    assert "[Previous Memory]" not in memory_update_prompt


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
        # By the second model call, the action step has been appended.
        assert len(state_arg["trajectory"]) == 1
        assert [m["role"] for m in state_arg["trajectory"][0]["completion"]] == [
            "assistant"
        ]
        assert state_arg["trajectory"][0]["extras"]["glyphbench_step_role"] == "action"
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
    # One env turn → two trajectory steps; both completions are assistant-only.
    assert len(state["trajectory"]) == 2
    assert state["num_turns"] == 1
    assert state["memory"] == "moved east"
    assert state["stop_condition"] == "has_final_env_response"
    assert [m["role"] for m in state["trajectory"][0]["completion"]] == ["assistant"]
    assert [m["role"] for m in state["trajectory"][1]["completion"]] == ["assistant"]
    assert state["trajectory"][0]["extras"]["glyphbench_step_role"] == "action"
    assert state["trajectory"][1]["extras"]["glyphbench_step_role"] == "memory"


@pytest.mark.asyncio
async def test_is_done_terminates_on_game_end():
    env = load_environment(task_id="glyphbench/__dummy-v0", num_episodes=1)
    state: dict = {"done": False}
    assert await env.is_done(state) is False
    state["done"] = True
    assert await env.is_done(state) is True


@pytest.mark.asyncio
async def test_memory_rollout_every_step_completion_is_assistant_only(monkeypatch):
    """prime-rl's pretokenize_rollout_trajectory asserts each step's
    completion is all-assistant. Run a 3-turn memory rollout and verify
    every trajectory step satisfies that invariant — both action steps and
    memory steps."""

    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        max_turns=3,
        use_memory=True,
    )
    row = env.dataset[0]
    call_idx = {"n": 0}

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        call_idx["n"] += 1
        # Action calls are odd; memory calls are even.
        if call_idx["n"] % 2 == 1:
            return _response(f"<think>t{call_idx['n']}</think><action>EAST</action>")
        return _response(f"<memory>turn-{call_idx['n']}</memory>")

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)

    state = await env.rollout(
        input=row,
        client=_NoopClient(object()),
        model="test-model",
        sampling_args={"max_tokens": 16},
    )

    # 3 env turns × 2 steps/turn = 6 trajectory steps.
    assert state["num_turns"] >= 1
    assert len(state["trajectory"]) == 2 * state["num_turns"]
    for i, step in enumerate(state["trajectory"]):
        roles = [m["role"] for m in step["completion"]]
        assert roles == ["assistant"], (
            f"step {i} ({step['extras'].get('glyphbench_step_role')!r}) "
            f"has non-assistant completion roles: {roles}"
        )
    # Even-indexed steps are action turns, odd-indexed are memory updates.
    for i, step in enumerate(state["trajectory"]):
        expected = "action" if i % 2 == 0 else "memory"
        assert step["extras"]["glyphbench_step_role"] == expected


def test_memory_mode_uses_full_memory_user_and_records_parse_flag(monkeypatch):
    """End-to-end memory-mode turn: action call generates a valid action,
    memory call generates malformed memory, expectation:
      - memory_user has all four sections incl. [Last Action]+[Next Observation]
      - state['memory'] retained (== '' on first turn since no prior)
      - memory step extras has memory_parse_failed=True
    """
    import asyncio
    from datasets import Dataset
    from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
    from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

    parser = GlyphbenchXMLParser()
    env = GlyphbenchMultiTurnEnv(
        dataset=Dataset.from_list([{
            "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
            "task": "glyphbench/__dummy-v0",
            "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            "answer": "",
        }]),
        rubric=EpisodicReturnRubric(parser=parser),
        parser=parser,
        n_frames=0,
        max_turns_override=None,
        max_output_tokens=128,
        use_memory=True,
        memory_update_max_tokens=64,
    )

    state = {
        "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
        "trajectory": [],
        "trajectory_id": "tid",
        "prompt": [],
        "sampling_args": {"max_tokens": 128},
    }
    asyncio.run(env.setup_state(state))

    valid_action = state["game"].action_spec.names[0]
    action_response = _response(
        f"<action>{valid_action}</action>",
        prompt_ids=[1, 2, 3],
        completion_ids=[10, 11],
    )

    # Stub get_model_response: action generates valid; memory generates malformed.
    async def fake_response(state, prompt_messages, sampling_args=None):
        memory_resp = _response(
            "I forgot to wrap my memory in tags",
            prompt_ids=[1, 2, 3, 10, 11, 20, 21],
            completion_ids=[30, 31],
        )
        return memory_resp

    env.get_model_response = fake_response  # type: ignore[assignment]
    asyncio.run(env.add_model_response(state, [], action_response))

    assert state["memory"] == ""  # previous memory retained (was empty initially)
    memory_step = state["trajectory"][-1]
    extras = memory_step["extras"]
    assert extras["glyphbench_step_role"] == "memory"
    assert extras["memory_parse_failed"] is True
    # Memory_user content check — four sections present.
    memory_prompt = memory_step["prompt"]
    memory_user_text = memory_prompt[-1]["content"]
    assert "[Last Action]" in memory_user_text
    assert "[Env Response]" in memory_user_text
    assert "[Next Observation]" in memory_user_text
    assert "[Memory Update]" in memory_user_text
    assert "[Previous Memory]" not in memory_user_text
    assert "[Action Response]" not in memory_user_text


def test_apply_action_response_forfeits_on_parse_fail(monkeypatch):
    """When parser reports parse_failed=True, env is NOT stepped; turn counter
    advances via forfeit_turn; action_chosen=='FORFEIT'; action_idx==-1; reward=0.
    """
    from datasets import Dataset
    from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
    from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

    parser = GlyphbenchXMLParser()
    env = GlyphbenchMultiTurnEnv(
        dataset=Dataset.from_list([{
            "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
            "task": "glyphbench/__dummy-v0",
            "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            "answer": "",
        }]),
        rubric=EpisodicReturnRubric(parser=parser),
        parser=parser,
        n_frames=0,
        max_turns_override=None,
        max_output_tokens=128,
        use_memory=False,
    )

    import asyncio
    state = {
        "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
        "trajectory": [],
        "trajectory_id": "tid",
        "prompt": [],
    }
    asyncio.run(env.setup_state(state))

    pre_turn = state["game"].turn
    pre_obs = state["current_obs"]
    messages = [
        {"role": "user", "content": pre_obs},
        {"role": "assistant", "content": "<think>I forgot to emit an action tag</think>"},
    ]
    result = env._apply_action_response(messages, state)

    assert result["parse_failed"] is True
    assert result["parse_failure_reason"] == "no_action_tag"
    assert result["action_chosen"] == "FORFEIT"
    assert result["action_idx"] == -1
    assert result["reward"] == 0.0
    assert result["forfeit"] is True
    assert state["game"].turn == pre_turn + 1
    assert state["forfeit_count"] == 1


def test_apply_action_response_steps_normally_on_valid_parse(monkeypatch):
    from datasets import Dataset
    from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
    from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

    parser = GlyphbenchXMLParser()
    env = GlyphbenchMultiTurnEnv(
        dataset=Dataset.from_list([{
            "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
            "task": "glyphbench/__dummy-v0",
            "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            "answer": "",
        }]),
        rubric=EpisodicReturnRubric(parser=parser),
        parser=parser,
        n_frames=0,
        max_turns_override=None,
        max_output_tokens=128,
        use_memory=False,
    )
    import asyncio
    state = {
        "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
        "trajectory": [],
        "trajectory_id": "tid",
        "prompt": [],
    }
    asyncio.run(env.setup_state(state))

    spec_names = state["game"].action_spec.names
    valid_action = spec_names[0]
    messages = [
        {"role": "user", "content": state["current_obs"]},
        {"role": "assistant", "content": f"<action>{valid_action}</action>"},
    ]
    result = env._apply_action_response(messages, state)
    assert result["parse_failed"] is False
    assert result["parse_failure_reason"] is None
    assert result["action_chosen"] == valid_action
    assert result["action_idx"] == 0
    assert result["forfeit"] is False


def test_state_accumulates_per_episode_counts():
    """Across an episode with mixed forfeits / truncations / memory parse fails,
    state[...] should expose count fields the rubric can divide.
    """
    import asyncio
    from datasets import Dataset
    from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
    from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

    parser = GlyphbenchXMLParser()
    env = GlyphbenchMultiTurnEnv(
        dataset=Dataset.from_list([{
            "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
            "task": "glyphbench/__dummy-v0",
            "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            "answer": "",
        }]),
        rubric=EpisodicReturnRubric(parser=parser),
        parser=parser,
        n_frames=0,
        max_turns_override=None,
        max_output_tokens=128,
        use_memory=False,
    )
    state = {
        "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
        "trajectory": [],
        "trajectory_id": "tid",
        "prompt": [],
    }
    asyncio.run(env.setup_state(state))

    assert state["forfeit_count"] == 0
    assert state["action_completion_truncations"] == 0
    assert state["memory_completion_truncations"] == 0
    assert state["memory_parse_failures"] == 0

    # Drive a forfeit through _apply_action_response; truncation/memory
    # counts are incremented by add_model_response when the response carries
    # is_truncated=True.
    messages = [
        {"role": "user", "content": state["current_obs"]},
        {"role": "assistant", "content": "no action tag here"},
    ]
    env._apply_action_response(messages, state)
    assert state["forfeit_count"] == 1


@pytest.mark.asyncio
async def test_memory_render_completion_stitches_split_steps(monkeypatch):
    """state['completion'] (the human-readable transcript) must look the
    same after the memory-step split as it did before — diff-based
    render_completion picks up only new messages from each step's prompt."""

    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        max_turns=2,
        use_memory=True,
    )
    row = env.dataset[0]
    call_idx = {"n": 0}

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        call_idx["n"] += 1
        if call_idx["n"] % 2 == 1:
            return _response(f"<think>t{call_idx['n']}</think><action>EAST</action>")
        return _response(f"<memory>turn-{call_idx['n']}</memory>")

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)

    state = await env.rollout(
        input=row,
        client=_NoopClient(object()),
        model="test-model",
        sampling_args={"max_tokens": 16},
    )

    full_transcript = list(state["prompt"]) + list(state["completion"])
    # Roles: starts with system + first user observation.
    assert full_transcript[0]["role"] == "system"
    assert full_transcript[1]["role"] == "user"

    # No two consecutive duplicate role/content pairs (i.e. the diff-based
    # stitcher didn't double-up the action message that lives in both
    # the action_step's completion and the memory_step's prompt).
    for a, b in zip(full_transcript, full_transcript[1:]):
        if a["role"] == b["role"] and a.get("content") == b.get("content"):
            raise AssertionError(
                f"render_completion produced consecutive duplicate messages: {a}"
            )


@pytest.mark.asyncio
async def test_action_completion_truncation_increments(monkeypatch):
    """add_model_response increments action_completion_truncations when the
    action Response has is_truncated=True (non-memory mode)."""
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        use_memory=False,
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
        "sampling_args": {"max_tokens": 16},
    }
    await env.setup_state(state)
    assert state["action_completion_truncations"] == 0

    # Truncated action response: finish_reason=length, is_truncated=True
    truncated_response = _response(
        "<action>EAST</action>",
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
        is_truncated=True,
    )
    await env.add_model_response(state, state["prompt"], truncated_response)
    assert state["action_completion_truncations"] == 1
    assert state["num_action_turns"] == 1


@pytest.mark.asyncio
async def test_memory_completion_truncation_increments(monkeypatch):
    """add_model_response increments memory_completion_truncations when the
    memory Response has is_truncated=True (memory mode)."""
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        use_memory=True,
        memory_update_max_tokens=8,
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
        "sampling_args": {"max_tokens": 16},
    }
    await env.setup_state(state)
    assert state["memory_completion_truncations"] == 0

    truncated_memory_response = _response(
        "<memory>truncated memory output",
        prompt_ids=[1, 2, 3, 4, 5, 6],
        completion_ids=[7, 8],
        is_truncated=True,
    )

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        return truncated_memory_response

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)

    action_response = _response(
        "<action>EAST</action>",
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
    )
    await env.add_model_response(state, state["prompt"], action_response)

    assert state["memory_completion_truncations"] == 1
    assert state["num_memory_turns"] == 1


@pytest.mark.asyncio
async def test_memory_parse_failure_increments(monkeypatch):
    """add_model_response increments memory_parse_failures when the
    memory Response contains no <memory>...</memory> tag (memory mode)."""
    env = load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        use_memory=True,
        memory_update_max_tokens=8,
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
        "sampling_args": {"max_tokens": 16},
    }
    await env.setup_state(state)
    assert state["memory_parse_failures"] == 0

    malformed_memory_response = _response(
        "I forgot to use memory tags",
        prompt_ids=[1, 2, 3, 4, 5, 6],
        completion_ids=[7, 8],
    )

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        return malformed_memory_response

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)

    action_response = _response(
        "<action>EAST</action>",
        prompt_ids=[1, 2],
        completion_ids=[3, 4],
    )
    await env.add_model_response(state, state["prompt"], action_response)

    assert state["memory_parse_failures"] == 1
    assert state["num_memory_turns"] == 1


@pytest.mark.asyncio
async def test_non_memory_action_step_extras_populated(monkeypatch):
    """Non-memory rollout: action-step extras are populated via env_response.

    Turn 0 — valid action (EAST): parse_failed=False, forfeit=False.
    Turn 1 — parse failure (no <action> tag): parse_failed=True, forfeit=True.

    We construct the env with no harness-level max_turns cap and set the
    game's own max_turns=2 so the episode terminates via truncation after
    turn 1 inside env_response — which is what populates extras on the last
    step.  Using the harness max_turns cap would cause max_turns_reached to
    fire before get_prompt_messages on iteration 3, skipping the env_response
    call that writes the extras.
    """
    from datasets import Dataset
    from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
    from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

    parser = GlyphbenchXMLParser()
    # max_turns_override=None → harness has no cap (max_turns=-1 in verifiers).
    # We set game.max_turns=2 post-setup so the episode truncates naturally
    # when env_response is called for iteration 3 (after step1 is appended),
    # ensuring env_response (and therefore extras population) runs for step1.
    env = GlyphbenchMultiTurnEnv(
        dataset=Dataset.from_list([{
            "info": '{"env_id": "glyphbench/__dummy-v0", "seed": 0}',
            "task": "glyphbench/__dummy-v0",
            "prompt": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
            "answer": "",
        }]),
        rubric=EpisodicReturnRubric(parser=parser),
        parser=parser,
        n_frames=0,
        max_turns_override=None,   # no harness-level cap
        max_output_tokens=128,
        use_memory=False,
    )

    call_idx = {"n": 0}
    game_holder: dict = {}
    responses = [
        # Turn 0: valid action
        _response("<think>go east</think><action>EAST</action>"),
        # Turn 1: parse failure — no <action> tag
        _response("<think>oops</think>I forgot the tag"),
    ]

    async def fake_get_model_response(state_arg, prompt_arg, **kwargs):
        # Capture the game on first call so we can set max_turns=2 before
        # the second LLM call begins.  After step0 is appended and before
        # get_prompt_messages for iteration 2 calls env_response, we need
        # the game to truncate at turn 2 (i.e. after the forfeit on turn 1).
        if call_idx["n"] == 0:
            game_holder["game"] = state_arg["game"]
            game_holder["game"].max_turns = 2
        resp = responses[call_idx["n"]]
        call_idx["n"] += 1
        return resp

    monkeypatch.setattr(env, "get_model_response", fake_get_model_response)

    row = env.dataset[0]
    state = await env.rollout(
        input=row,
        client=_NoopClient(object()),
        model="test-model",
        sampling_args={"max_tokens": 16},
    )

    traj = state["trajectory"]
    # Exactly 2 trajectory steps (2 LLM calls in non-memory mode).
    assert len(traj) == 2, f"expected 2 steps, got {len(traj)}"

    step0_extras = traj[0].get("extras") or {}
    assert step0_extras.get("glyphbench_step_role") == "action", step0_extras
    assert step0_extras.get("parse_failed") is False, step0_extras
    assert step0_extras.get("forfeit") is False, step0_extras
    assert step0_extras.get("action_chosen") == "EAST", step0_extras

    step1_extras = traj[1].get("extras") or {}
    assert step1_extras.get("glyphbench_step_role") == "action", step1_extras
    assert step1_extras.get("parse_failed") is True, step1_extras
    assert step1_extras.get("forfeit") is True, step1_extras
    assert step1_extras.get("action_chosen") == "FORFEIT", step1_extras
