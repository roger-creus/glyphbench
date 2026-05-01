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
    action_text: str,
    action_chosen: str,
    parse_failed: bool,
    parse_failure_reason: str | None,
    action_truncated: bool,
    reward: float,
    terminated: bool,
    truncated: bool,
    next_obs: str,
) -> vf.UserMessage:
    """Build the user message for the memory-update generation.

    Sections, in order:

      [Last Action]      — your previous response (raw reasoning + action
                           tag) plus the parser's outcome (action applied
                           or forfeit + reason) and an output-truncated
                           flag. The chat template strips ``<think>`` from
                           prior assistant turns by default, so this block
                           re-injects the reasoning as plain text — gives
                           the memory writer continuity with the action
                           turn's analysis instead of forcing it to redo
                           the work from scratch.

      [Env Response]     — reward, terminated, truncated (episode-level).

      [Next Observation] — the new grid + legend + HUD + message produced
                           by the env's response to the action. Not
                           privileged: the agent will see this same view
                           at the next action turn anyway. Surfacing it
                           here lets memory react to the action's actual
                           effect, which a sparse-reward signal alone
                           cannot convey.

      [Memory Update]    — write instruction with the synth/non-describe
                           guidance.
    """
    if parse_failed:
        parse_status = (
            f"FAILED — {parse_failure_reason or 'no <action> tag'} "
            "(turn forfeited; env state did not advance)"
        )
    else:
        parse_status = "ok"

    last_action_text = action_text.strip() if action_text else "(empty)"

    last_action = (
        "[Last Action]\n"
        "Your previous response (raw):\n"
        "---\n"
        f"{last_action_text}\n"
        "---\n"
        "Outcome:\n"
        f"  Action applied: {action_chosen}\n"
        f"  Parse status: {parse_status}\n"
        f"  Output truncated: {str(bool(action_truncated)).lower()}"
    )

    env_response_block = (
        "[Env Response]\n"
        f"  Reward: {reward:+.3f}\n"
        f"  Terminated: {str(bool(terminated)).lower()}\n"
        f"  Truncated: {str(bool(truncated)).lower()}"
    )

    next_obs_clean = next_obs.rstrip() if next_obs else "(none)"
    next_obs_block = (
        "[Next Observation]\n"
        f"{next_obs_clean}"
    )

    instruction = (
        "[Memory Update]\n"
        "Update your memory based on the above: the action you took, the "
        "env's response, and the new observation. Reply ONLY with "
        "`<memory>...your concise updated memory...</memory>`. Anything "
        "outside the <memory> tag is discarded. Do not emit an <action> "
        "tag. Memory is for synthesis — causal facts you've established, "
        "plans, discoveries, retracted hypotheses, what the last action "
        "actually did vs. what you expected. Do NOT re-describe the grid: "
        "the next observation above is shown again at the next action "
        "turn, so duplicating its contents wastes the budget."
    )

    content = "\n\n".join(
        [last_action, env_response_block, next_obs_block, instruction]
    )
    return vf.UserMessage(content=content)


def action_response_text(action_completion: list[Any]) -> str:
    """Stitch the action turn's full text from a vf assistant message list.

    Qwen3.5's chat template prefills ``<think>\\n`` and emits
    ``reasoning_content`` separately from ``content``. To preserve the
    full reasoning trace for the memory-turn re-injection, we glue them
    back together with explicit ``<think>`` delimiters when reasoning is
    set; otherwise the raw content already contains everything (for
    older / non-thinking responses, or for chat templates that don't
    split out the reasoning).
    """
    if not action_completion:
        return ""
    msg = action_completion[-1]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content") or ""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n{content}".strip()
    return content.strip()


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
