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
from typing import Iterable

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
    "ACTION_NAME must be one of the names in the per-turn [Actions] list. "
    "Read the grid directly — the player glyph shows your position and "
    "orientation. Total response budget: {budget} tokens (reasoning + "
    "action combined). Anything outside the <action> tag is ignored. If "
    "the <action> tag is missing or contains an unknown name, the {noop} "
    "action is applied automatically."
)


def build_system_prompt(game: BaseGlyphEnv, max_output_tokens: int) -> str:
    """Compose the system prompt: game rules + standard response-format block.

    The output is stable across turns — verifiers reuses the cached tokenisation
    as long as this content doesn't change.
    """
    header = game.system_prompt().rstrip()
    fmt = RESPONSE_FORMAT_BLOCK_TMPL.format(
        budget=max_output_tokens,
        noop=game.noop_action_name,
    )
    # Most env classes already append `action_spec.render_for_prompt()` in
    # their own `system_prompt()` (see e.g. minigrid/base.py, procgen/base.py,
    # most atari/*.py). Only inject an extra Actions block here when the
    # header doesn't already contain it — avoids the duplicate "Actions"
    # listing the user flagged on 2026-04-25.
    actions_block = game.action_spec.render_for_prompt().strip()
    if actions_block and actions_block not in header:
        return f"{header}\n\n---\n{actions_block}\n\n---\n{fmt}"
    return f"{header}\n\n---\n{fmt}"


def render_user_turn(
    game: BaseGlyphEnv,
    frames: deque[tuple[str, str, float]] | Iterable[tuple[str, str, float]],
    current_obs: str,
    turn: int,
    max_output_tokens: int,
) -> str:
    """Render the user-turn message for the current timestep.

    Args:
        game: the live game instance (used for the action list).
        frames: iterable of (obs_text_before_action, action_name, reward) tuples
                in temporal order — oldest first, newest last.
        current_obs: the full ``GridObservation.render()`` text for this turn.
        turn: the absolute turn number (``game.turn`` after the last step).
        max_output_tokens: per-turn LLM budget, echoed in the footer reminder.

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
    if legend_lines:
        parts.append("[Legend]\n" + "\n".join(legend_lines))

    # 2. History block (only if non-empty).
    if frames_list:
        parts.append(_render_history(frames_list, turn))

    # 3. Current observation — strip legend, keep HUD + grid + message.
    parts.append(_render_current_block(current_obs, turn))

    # 4. Actions enumerated (names-only — full descriptions are in the
    # stable system prompt; this is just a per-turn reminder).
    action_list = ", ".join(game.action_spec.names)
    parts.append(f"[Actions]\nChoose one: [{action_list}]")

    # 5. Per-turn format reminder. Stays short — the full response-format
    # rules are in the system prompt; this just nudges the model to emit
    # the action tag now.
    parts.append(f"Now emit your move as `<action>ACTION_NAME</action>`.")

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
        lines.append(
            f"(turn {past_turn})\n"
            f"{grid}\n"
            f"chose {action} → reward {reward:+.3f}".replace("  \n", "").replace("\n\n", "\n")
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
