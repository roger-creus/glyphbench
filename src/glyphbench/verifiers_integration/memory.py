"""Memory-turn helpers for the glyphbench verifiers integration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from verifiers.types import SamplingArgs, TrajectoryStepTokens

_MEMORY_RE = re.compile(
    r"<\s*memory\s*>(.*?)<\s*/\s*memory\s*>", re.DOTALL | re.IGNORECASE
)
_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class MemoryExtraction:
    memory: str
    mode: str


def extract_memory_update(text: str) -> MemoryExtraction:
    """Extract stored memory from a memory-update assistant response."""
    raw = text or ""
    tag_matches = _MEMORY_RE.findall(raw)
    if tag_matches:
        tagged = [m.strip() for m in tag_matches if m.strip()]
        return MemoryExtraction(memory="\n\n".join(tagged), mode="tag")

    close_matches = list(_THINK_CLOSE_RE.finditer(raw))
    if close_matches:
        memory = raw[close_matches[-1].end() :].strip()
        return MemoryExtraction(memory=memory, mode="post_think")

    open_match = _THINK_OPEN_RE.search(raw)
    if open_match:
        memory = raw[: open_match.start()].strip()
        return MemoryExtraction(memory=memory, mode="unterminated_think")

    memory = _THINK_TAG_RE.sub("", raw).strip()
    return MemoryExtraction(memory=memory, mode="stripped_text")


def build_memory_update_user(
    *,
    previous_memory: str,
    action_response: str,
    parsed_action: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    next_observation: str,
) -> vf.UserMessage:
    """Build the second user message for a memory-enabled environment step."""
    content = (
        "[Memory Update]\n"
        "Update the memory that will be shown on the next environment turn. "
        "Keep it concise.\n\n"
        "[Previous Memory]\n"
        "<memory>\n"
        f"{previous_memory or ''}\n"
        "</memory>\n\n"
        "[Action Response]\n"
        f"{action_response or ''}\n\n"
        "[Environment Feedback]\n"
        f"Parsed action: {parsed_action}\n"
        f"Reward: {reward:+.3f}\n"
        f"Terminated: {str(bool(terminated)).lower()}\n"
        f"Truncated: {str(bool(truncated)).lower()}\n\n"
        "[Next Observation]\n"
        f"{next_observation}\n\n"
        "Write the updated memory."
    )
    return vf.UserMessage(content=content)


def memory_sampling_args(
    sampling_args: SamplingArgs | None,
    memory_update_max_tokens: int | None,
) -> SamplingArgs | None:
    """Return sampling args for the memory-update generation."""
    if memory_update_max_tokens is None:
        return None
    args: dict[str, Any] = dict(sampling_args or {})
    if "max_completion_tokens" in args:
        args["max_completion_tokens"] = int(memory_update_max_tokens)
    elif "max_tokens" in args:
        args["max_tokens"] = int(memory_update_max_tokens)
    else:
        args["max_tokens"] = int(memory_update_max_tokens)
    return args


def merge_memory_step_tokens(
    *,
    action_tokens: TrajectoryStepTokens | None,
    memory_tokens: TrajectoryStepTokens | None,
) -> TrajectoryStepTokens | None:
    """Merge action and memory-update token records into one trainable step.

    The memory-update prompt must start with the exact action prompt and action
    completion token prefix. Tokens after that prefix are the memory-update
    user prompt bridge: visible conditioning context with loss mask 0.
    """
    if action_tokens is None or memory_tokens is None:
        return None

    prefix = action_tokens["prompt_ids"] + action_tokens["completion_ids"]
    memory_prompt_ids = memory_tokens["prompt_ids"]
    if memory_prompt_ids[: len(prefix)] != prefix:
        return None

    bridge_ids = memory_prompt_ids[len(prefix) :]
    bridge_len = len(bridge_ids)
    routed_experts = action_tokens.get("routed_experts")
    memory_routed = memory_tokens.get("routed_experts")
    if routed_experts is not None or memory_routed is not None:
        routed_experts = (routed_experts or []) + ([[]] * bridge_len) + (
            memory_routed or []
        )

    return TrajectoryStepTokens(
        prompt_ids=action_tokens["prompt_ids"],
        prompt_mask=action_tokens["prompt_mask"],
        completion_ids=(
            action_tokens["completion_ids"]
            + bridge_ids
            + memory_tokens["completion_ids"]
        ),
        completion_mask=(
            action_tokens["completion_mask"]
            + [0] * bridge_len
            + memory_tokens["completion_mask"]
        ),
        completion_logprobs=(
            action_tokens["completion_logprobs"]
            + [0.0] * bridge_len
            + memory_tokens["completion_logprobs"]
        ),
        overlong_prompt=bool(
            action_tokens.get("overlong_prompt")
            or memory_tokens.get("overlong_prompt")
        ),
        is_truncated=bool(
            action_tokens.get("is_truncated") or memory_tokens.get("is_truncated")
        ),
        routed_experts=routed_experts,
    )
