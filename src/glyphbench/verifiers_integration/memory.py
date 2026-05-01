"""Memory-turn helpers for the glyphbench verifiers integration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from verifiers.types import SamplingArgs

@dataclass(frozen=True)
class MemoryExtraction:
    """Result of parsing the memory-turn assistant response."""
    memory: str
    parse_failed: bool


_MEMORY_RE = re.compile(
    r"<\s*memory\s*>(.*?)<\s*/\s*memory\s*>", re.DOTALL | re.IGNORECASE
)


def extract_memory_update(text: str) -> MemoryExtraction:
    """Strict <memory>...</memory> extraction.

    Returns ``MemoryExtraction(memory=joined_tags, parse_failed=False)`` if at
    least one complete ``<memory>...</memory>`` tag is found (multiple tags
    are joined with a blank-line separator). Anything else — no tag, only an
    open tag, raw post-think text — yields ``MemoryExtraction(memory="",
    parse_failed=True)``.

    The caller (verifiers integration) keeps the previous memory on failure.
    """
    raw = text or ""
    matches = _MEMORY_RE.findall(raw)
    if not matches:
        return MemoryExtraction(memory="", parse_failed=True)
    return MemoryExtraction(memory="\n\n".join(m.strip() for m in matches), parse_failed=False)


def build_memory_update_user(
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> vf.UserMessage:
    """Build the lean user message for the memory-update generation.

    The memory writer already sees, via the conversation prefix:
      - previous memory (in user_obs_t's [Memory] block)
      - the obs that prompted the action (user_obs_t's [Grid]/[Legend]/[Message])
      - the action chosen (assistant_action_t's <action> tag; <think> elided
        by the chat template under enable_thinking=False)

    This wrapper only adds the env's response (reward + termination flags)
    and the write instruction. No future peeking, no duplicates.
    """
    content = (
        "[Memory Update]\n"
        "The environment responded to your last action with:\n"
        f"  Reward: {reward:+.3f}\n"
        f"  Terminated: {str(bool(terminated)).lower()}\n"
        f"  Truncated: {str(bool(truncated)).lower()}\n\n"
        "Update your memory based on this turn's outcome and the observation "
        "above. Write the updated memory inside <memory>...</memory> tags. "
        "Anything outside the <memory> tag is discarded. Do not emit an "
        "<action> tag in this response."
    )
    return vf.UserMessage(content=content)


def memory_sampling_args(
    sampling_args: SamplingArgs | None,
    memory_update_max_tokens: int | None,
) -> SamplingArgs | None:
    """Return sampling args for the memory-update generation.

    Always disables thinking-mode for the memory-update call when an
    `extra_body.chat_template_kwargs` slot is present (Qwen3/Qwen3.5).
    The action turn's RESPONSE FORMAT block conditions the model to emit
    ``<action>...</action>`` after ``</think>``; if we left thinking on for
    the memory update the model would default to that habit and our parser
    would store the action tag as the memory.
    """
    if memory_update_max_tokens is None:
        return None
    args: dict[str, Any] = dict(sampling_args or {})
    if "max_completion_tokens" in args:
        args["max_completion_tokens"] = int(memory_update_max_tokens)
    elif "max_tokens" in args:
        args["max_tokens"] = int(memory_update_max_tokens)
    else:
        args["max_tokens"] = int(memory_update_max_tokens)
    extra_body = dict(args.get("extra_body") or {})
    chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_kwargs
    args["extra_body"] = extra_body
    return args
