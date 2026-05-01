"""Prompt construction for the glyphbench verifiers integration.

Two entry points:
    build_system_prompt(game, max_output_tokens) → str
        Composed from game.system_prompt() + a standardised response-format
        block. Called ONCE per rollout (static across turns).

    render_user_turn(game, frames, current_obs, turn) → str
        The per-turn user message. Layout:

            [Legend]
              <union of glyphs across frames + current, rendered once>

            [History — last N turns]                 (omitted entirely if N=0)
              (turn T-K) <grid-only view>
                  <one-line HUD delta>
                chose ACTION → reward R

              (turn T-K+1) ...

            [Current Observation — turn T]
              <full grid + HUD + optional message>

            [Actions]
              Choose one: [A, B, C, ...]

            Respond with <think>...</think><action>NAME</action>.
            Total response budget: 512 tokens.

Design goals:
    * One legend per user message, deduped across history + current.
    * Stable section order → KV-cache prefix overlap across turns.
    * History frames stripped of legend + HUD-delta-only, saving tokens.
    * Current observation kept intact (full grid + HUD + message), minus legend.
    * Budget reminded every turn, deters runaway thinking.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable

from glyphbench.core.base_env import BaseGlyphEnv

_LEGEND_RE = re.compile(r"\[Legend\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_GRID_RE = re.compile(r"\[Grid\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_HUD_RE = re.compile(r"\[HUD\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_MESSAGE_RE = re.compile(r"\[Message\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)


RESPONSE_FORMAT_BLOCK_TMPL = (
    "RESPONSE FORMAT\n"
    "Pick the next move and emit exactly one tag:\n"
    "  <action>ACTION_NAME</action>\n"
    "\n"
    "ACTION_NAME must be one of the names in the [Actions] list. Read the "
    "grid directly — the player glyph shows your position and orientation. "
    "Anything outside the <action> tag is ignored.\n"
    "\n"
    "Output budget: {budget} tokens (reasoning + action combined). Be concise — "
    "if your full response exceeds {budget} tokens before the closing "
    "</action> tag, the action will be discarded and the turn forfeited "
    "(env state unchanged, turn counter still advances). If the <action> tag "
    "is missing or contains an unknown name, the turn is also forfeited."
)

OBSERVATION_CONVENTIONS_BLOCK = (
    "OBSERVATION CONVENTIONS\n"
    "Each turn you receive a [Grid] (ASCII), a [Legend] mapping each glyph "
    "to its meaning for this turn, and a [HUD] line that always contains "
    "`Step: T / N` (current turn / per-episode budget). Optionally a "
    "[Message] block with one-shot env feedback (last action result, etc.). "
    "Treat the [Grid] as authoritative for spatial state and the [Legend] "
    "as authoritative for glyph meaning. The full action list is in this "
    "system prompt and does not repeat per turn."
)

MEMORY_BLOCK_TMPL = (
    "MEMORY MODE (active for this run)\n"
    "Each environment turn is followed by a memory-update turn. After "
    "you emit the action above, you'll get a [Memory Update] prompt "
    "containing your previous memory, your action response, the env's "
    "feedback, and the next observation. Reply ONLY with "
    "`<memory>...your concise updated memory...</memory>` — do not emit "
    "an <action> tag in the memory turn. The text inside <memory> is "
    "carried into the next turn's observation as a [Memory] block "
    "(authoritative source for state you want to track across turns; the "
    "current observation still wins on conflicts). Cap the memory at "
    "{memory_budget} tokens; anything beyond will be truncated."
)


def build_system_prompt(
    game: BaseGlyphEnv,
    max_output_tokens: int,
    *,
    use_memory: bool = False,
    memory_update_max_tokens: int | None = None,
) -> str:
    """Compose the system prompt: game rules + standard response-format
    block + observation conventions (+ memory block when memory mode is
    on).

    The output is stable across turns — verifiers reuses the cached
    tokenisation as long as this content doesn't change.
    """
    header = game.system_prompt().rstrip()
    fmt = RESPONSE_FORMAT_BLOCK_TMPL.format(budget=max_output_tokens)
    blocks: list[str] = [header, OBSERVATION_CONVENTIONS_BLOCK]
    # Most env classes already append `action_spec.render_for_prompt()` in
    # their own `system_prompt()` (see e.g. minigrid/base.py, procgen/base.py,
    # most atari/*.py). Only inject an extra Actions block here when the
    # header doesn't already contain it — avoids the duplicate "Actions"
    # listing the user flagged on 2026-04-25.
    actions_block = game.action_spec.render_for_prompt().strip()
    if actions_block and actions_block not in header:
        blocks.append(actions_block)
    blocks.append(fmt)
    if use_memory:
        blocks.append(MEMORY_BLOCK_TMPL.format(
            memory_budget=memory_update_max_tokens or 4096,
        ))
    return "\n\n---\n".join(blocks)


def render_user_turn(
    game: BaseGlyphEnv,
    frames: deque[tuple[str, str, float]] | Iterable[tuple[str, str, float]],
    current_obs: str,
    turn: int,
    max_output_tokens: int,
    *,
    memory: str | None = None,
) -> str:
    """Render the user-turn message for the current timestep.

    Args:
        game: the live game instance (used for the action list).
        frames: iterable of (obs_text_before_action, action_name, reward) tuples
                in temporal order — oldest first, newest last.
        current_obs: the full ``GridObservation.render()`` text for this turn.
        turn: the absolute turn number (``game.turn`` after the last step).
        max_output_tokens: per-turn LLM budget, echoed in the footer reminder.
        memory: optional carried state to show before the current observation.

    Returns:
        The user-turn string.
    """
    frames_list = list(frames)

    # 1. Build merged legend across history + current.
    legend_lines: dict[str, None] = {}  # ordered-set via dict
    for obs, _, _ in frames_list:
        for line in _extract_legend_lines(obs):
            legend_lines.setdefault(line, None)
    for line in _extract_legend_lines(current_obs):
        legend_lines.setdefault(line, None)

    parts: list[str] = []
    if memory is not None:
        parts.append(
            "[Memory]\n"
            "Use this as carried state from previous turns. The current "
            "observation is authoritative if it conflicts.\n\n"
            "<memory>\n"
            f"{memory}\n"
            "</memory>"
        )
    if legend_lines:
        parts.append("[Legend]\n" + "\n".join(legend_lines))

    # 2. History block (only if non-empty).
    if frames_list:
        parts.append(_render_history(frames_list, turn))

    # 3. Current observation — strip legend, keep HUD + grid + message.
    parts.append(_render_current_block(current_obs, turn))

    # 4. Per-turn nudge — the full action list with descriptions is in
    # the cached system prompt, so we don't repeat it here. This line
    # only re-asserts the response format the agent must use right now.
    parts.append("Now emit your move as `<action>ACTION_NAME</action>`.")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_legend_lines(obs: str) -> list[str]:
    m = _LEGEND_RE.search(obs)
    if not m:
        return []
    return [ln for ln in m.group(1).splitlines() if ln.strip()]


def _extract_grid(obs: str) -> str:
    m = _GRID_RE.search(obs)
    return m.group(1).rstrip() if m else ""


def _extract_hud(obs: str) -> str:
    m = _HUD_RE.search(obs)
    return m.group(1).strip() if m else ""


def _extract_message(obs: str) -> str:
    m = _MESSAGE_RE.search(obs)
    return m.group(1).strip() if m else ""


def _render_history(frames_list: list[tuple[str, str, float]], current_turn: int) -> str:
    n = len(frames_list)
    lines = [f"[History — last {n} turn{'s' if n != 1 else ''}]"]
    # Number each historical turn from T-N .. T-1 (T is the current turn).
    # Deliberately omit the [HUD] block — it leaks privileged state
    # (positions, counts, intents) that the agent must instead read off the
    # Unicode glyphs. See the user policy in DECISIONS.md (HUD ban).
    for i, (obs, action, reward) in enumerate(frames_list):
        past_turn = current_turn - (n - i)
        grid = _extract_grid(obs)
        if action == "FORFEIT":
            action_line = "chose FORFEIT (parse failed) → reward 0"
        else:
            action_line = f"chose {action} → reward {reward:+.3f}"
        lines.append(
            f"(turn {past_turn})\n"
            f"{grid}\n"
            f"{action_line}".replace("  \n", "").replace("\n\n", "\n")
        )
    return "\n".join(lines)


def _render_current_block(current_obs: str, turn: int) -> str:
    # Drop legend (rendered globally above) and HUD (per design: every
    # game-relevant fact must be readable off the Unicode grid). Envs
    # may still compute a HUD for their own internal use (debugging,
    # info-dict reporting), but it is NOT shown to the model.
    grid = _extract_grid(current_obs)
    msg = _extract_message(current_obs)
    parts = [f"[Current Observation — turn {turn}]"]
    if grid:
        parts.append(f"[Grid]\n{grid}")
    if msg:
        parts.append(f"[Message]\n{msg}")
    return "\n".join(parts)
