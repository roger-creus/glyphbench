"""Tests for memory-turn helpers."""

from __future__ import annotations

from glyphbench.verifiers_integration.memory import (
    action_response_text,
    build_memory_update_user,
    extract_memory_update,
    memory_sampling_args,
)


_OBS = (
    "[Legend]\nA — agent\n\n"
    "[HUD]\nStep: 1 / 50\n\n"
    "[Grid]\n.A.\n...\n\n"
    "[Message]\nmoved east"
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


def test_build_memory_update_user_includes_all_four_sections():
    msg = build_memory_update_user(
        action_text="<think>go east</think>\n<action>EAST</action>",
        action_chosen="EAST",
        parse_failed=False,
        parse_failure_reason=None,
        action_truncated=False,
        reward=1.0,
        terminated=True,
        truncated=False,
        next_obs=_OBS,
    )
    text = msg["content"]
    # Section headers
    assert "[Last Action]" in text
    assert "[Env Response]" in text
    assert "[Next Observation]" in text
    assert "[Memory Update]" in text
    # Last Action: raw response and parsed outcome
    assert "<think>go east</think>" in text
    assert "<action>EAST</action>" in text
    assert "Action applied: EAST" in text
    assert "Parse status: ok" in text
    assert "Output truncated: false" in text
    # Env Response
    assert "Reward: +1.000" in text
    assert "Terminated: true" in text
    assert "Truncated: false" in text
    # Next Observation: actual grid is inlined
    assert "[Grid]\n.A.\n..." in text
    # Memory Update instruction
    assert "<memory>" in text and "</memory>" in text
    assert "Anything outside the <memory> tag is discarded" in text
    assert "Do not emit an <action> tag" in text
    # Prompt steers toward synthesis, not grid description
    assert "Do NOT re-describe the grid" in text


def test_build_memory_update_user_renders_parse_failure_explicitly():
    msg = build_memory_update_user(
        action_text="i forgot the action tag",
        action_chosen="FORFEIT",
        parse_failed=True,
        parse_failure_reason="no_action_tag",
        action_truncated=False,
        reward=0.0,
        terminated=False,
        truncated=False,
        next_obs=_OBS,
    )
    text = msg["content"]
    assert "Action applied: FORFEIT" in text
    assert "Parse status: FAILED" in text
    assert "no_action_tag" in text
    assert "turn forfeited" in text


def test_build_memory_update_user_flags_output_truncation():
    msg = build_memory_update_user(
        action_text="<think>thinking",
        action_chosen="FORFEIT",
        parse_failed=True,
        parse_failure_reason="no_action_tag",
        action_truncated=True,
        reward=0.0,
        terminated=False,
        truncated=False,
        next_obs=_OBS,
    )
    text = msg["content"]
    assert "Output truncated: true" in text


def test_build_memory_update_user_handles_negative_reward_and_truncation():
    msg = build_memory_update_user(
        action_text="<action>NORTH</action>",
        action_chosen="NORTH",
        parse_failed=False,
        parse_failure_reason=None,
        action_truncated=False,
        reward=-0.5,
        terminated=False,
        truncated=True,
        next_obs=_OBS,
    )
    text = msg["content"]
    assert "Reward: -0.500" in text
    assert "Terminated: false" in text
    assert "Truncated: true" in text


def test_action_response_text_stitches_reasoning_when_separate():
    completion = [{
        "role": "assistant",
        "content": "<action>EAST</action>",
        "reasoning_content": "the goal is east",
    }]
    out = action_response_text(completion)
    assert "<think>" in out
    assert "the goal is east" in out
    assert "</think>" in out
    assert "<action>EAST</action>" in out


def test_action_response_text_uses_content_when_no_reasoning_field():
    completion = [{
        "role": "assistant",
        "content": "thinking</think><action>EAST</action>",
        "reasoning_content": None,
    }]
    out = action_response_text(completion)
    assert out == "thinking</think><action>EAST</action>"


def test_action_response_text_handles_empty_completion():
    assert action_response_text([]) == ""


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
