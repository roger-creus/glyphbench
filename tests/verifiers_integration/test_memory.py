"""Tests for memory-turn helpers."""

from __future__ import annotations

from glyphbench.verifiers_integration.memory import (
    build_memory_update_user,
    extract_memory_update,
    memory_sampling_args,
    merge_memory_step_tokens,
)


def test_extract_memory_returns_tag_content_when_present():
    result = extract_memory_update("<memory>A</memory>")
    assert result.memory == "A"
    assert result.parse_failed is False


def test_extract_memory_concatenates_multiple_tags():
    result = extract_memory_update(
        "<think>x</think><memory>A</memory><memory>B</memory>"
    )
    assert result.memory == "A\n\nB"
    assert result.parse_failed is False


def test_extract_memory_empty_tag_is_intentional_empty():
    result = extract_memory_update("<memory></memory>")
    assert result.memory == ""
    assert result.parse_failed is False


def test_extract_memory_no_tag_signals_parse_failure():
    result = extract_memory_update("<think>draft</think>keep this")
    assert result.parse_failed is True
    assert result.memory == ""


def test_extract_memory_unterminated_tag_signals_parse_failure():
    result = extract_memory_update("<memory>open but never closed")
    assert result.parse_failed is True
    assert result.memory == ""


def test_extract_memory_empty_string_signals_parse_failure():
    result = extract_memory_update("")
    assert result.parse_failed is True
    assert result.memory == ""


def test_build_memory_update_user_is_lean():
    msg = build_memory_update_user(
        reward=1.0,
        terminated=True,
        truncated=False,
    )
    text = msg["content"]
    # Required content
    assert "[Memory Update]" in text
    assert "Reward: +1.000" in text
    assert "Terminated: true" in text
    assert "Truncated: false" in text
    assert "<memory>" in text and "</memory>" in text
    assert "Anything outside the <memory> tag is discarded" in text
    assert "Do not emit an <action> tag" in text
    # Removed content (these MUST be absent — already shown elsewhere
    # in the conversation, or future-peeking)
    assert "[Previous Memory]" not in text
    assert "[Action Response]" not in text
    assert "[Next Observation]" not in text
    assert "Parsed action:" not in text


def test_build_memory_update_user_handles_negative_reward_and_truncation():
    msg = build_memory_update_user(reward=-0.5, terminated=False, truncated=True)
    text = msg["content"]
    assert "Reward: -0.500" in text
    assert "Terminated: false" in text
    assert "Truncated: true" in text


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
