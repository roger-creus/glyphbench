"""Tests for memory-turn helpers."""

from __future__ import annotations

from glyphbench.verifiers_integration.memory import (
    build_memory_update_user,
    extract_memory_update,
    memory_sampling_args,
    merge_memory_step_tokens,
)


def test_extract_memory_prefers_memory_tags():
    result = extract_memory_update(
        "<think>x</think><memory>A</memory><memory>B</memory>"
    )
    assert result.memory == "A\n\nB"
    assert result.mode == "tag"


def test_extract_memory_empty_memory_tags_are_intentional_empty_memory():
    result = extract_memory_update(
        "<think>x</think><memory></memory>ignore this outside tag"
    )
    assert result.memory == ""
    assert result.mode == "tag"


def test_extract_memory_uses_text_after_final_think():
    result = extract_memory_update("<think>draft</think>keep this")
    assert result.memory == "keep this"
    assert result.mode == "post_think"


def test_extract_memory_ignores_unterminated_thinking():
    result = extract_memory_update("<think>draft remember north wall")
    assert result.memory == ""
    assert result.mode == "unterminated_think"


def test_extract_memory_accepts_plain_text_without_tags():
    result = extract_memory_update("remember north wall")
    assert result.memory == "remember north wall"
    assert result.mode == "stripped_text"


def test_build_memory_update_user_mentions_feedback():
    msg = build_memory_update_user(
        previous_memory="door at east",
        action_response="<think>try</think><action>EAST</action>",
        parsed_action="EAST",
        reward=1.0,
        terminated=True,
        truncated=False,
        next_observation="[Grid]\nG",
    )
    text = msg["content"]
    assert "Keep it concise" in text
    assert "2048" not in text
    assert "door at east" in text
    assert "<think>try</think><action>EAST</action>" in text
    assert "Parsed action: EAST" in text
    assert "Reward: +1.000" in text
    assert "Terminated: true" in text
    assert "Truncated: false" in text
    assert "[Grid]\nG" in text
    assert "<memory>" in text and "</memory>" in text
    assert "Do not emit an <action> tag" in text


def test_memory_sampling_args_defaults_to_existing_action_args():
    assert memory_sampling_args({"max_tokens": 32}, None) is None


def test_memory_sampling_args_overrides_existing_limit_name():
    expected_no_think = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    assert memory_sampling_args({"max_completion_tokens": 32}, 7) == {
        "max_completion_tokens": 7,
        **expected_no_think,
    }
    assert memory_sampling_args({"max_tokens": 32}, 9) == {
        "max_tokens": 9,
        **expected_no_think,
    }
    assert memory_sampling_args({}, 11) == {
        "max_tokens": 11,
        **expected_no_think,
    }


def test_memory_sampling_args_disables_thinking_when_action_args_have_it():
    action_args = {
        "max_tokens": 8192,
        "temperature": 1.0,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    out = memory_sampling_args(action_args, 4096)
    assert out["max_tokens"] == 4096
    assert out["temperature"] == 1.0
    extra = out["extra_body"]
    assert extra["top_k"] == 20
    assert extra["min_p"] == 0.0
    assert extra["chat_template_kwargs"]["enable_thinking"] is False


def test_merge_tokens_masks_memory_update_user_bridge():
    merged = merge_memory_step_tokens(
        action_tokens={
            "prompt_ids": [1, 2],
            "prompt_mask": [0, 0],
            "completion_ids": [3, 4],
            "completion_mask": [1, 1],
            "completion_logprobs": [-0.3, -0.4],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": None,
        },
        memory_tokens={
            "prompt_ids": [1, 2, 3, 4, 5, 6],
            "prompt_mask": [0, 0, 0, 0, 0, 0],
            "completion_ids": [7, 8],
            "completion_mask": [1, 1],
            "completion_logprobs": [-0.7, -0.8],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": None,
        },
    )

    assert merged is not None
    assert merged["prompt_ids"] == [1, 2]
    assert merged["completion_ids"] == [3, 4, 5, 6, 7, 8]
    assert merged["completion_mask"] == [1, 1, 0, 0, 1, 1]
    assert merged["completion_logprobs"] == [-0.3, -0.4, 0.0, 0.0, -0.7, -0.8]


def test_merge_tokens_returns_none_on_prefix_mismatch():
    merged = merge_memory_step_tokens(
        action_tokens={
            "prompt_ids": [1],
            "prompt_mask": [0],
            "completion_ids": [2],
            "completion_mask": [1],
            "completion_logprobs": [-0.2],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": None,
        },
        memory_tokens={
            "prompt_ids": [1, 99, 3],
            "prompt_mask": [0, 0, 0],
            "completion_ids": [4],
            "completion_mask": [1],
            "completion_logprobs": [-0.4],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": None,
        },
    )
    assert merged is None
