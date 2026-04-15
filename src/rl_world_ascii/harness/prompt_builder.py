"""Build the per-turn user prompt from EpisodeState and the current observation.

The system prompt is rendered separately via `templating.render_system_prompt`.
This module only produces the user message that changes each turn.
"""

from __future__ import annotations

from rl_world_ascii.core.observation import GridObservation
from rl_world_ascii.harness.state import EpisodeState


def build_user_prompt(
    state: EpisodeState,
    current_obs: GridObservation,
    *,
    turn_index: int,
) -> str:
    """Return the user-message string for this turn.

    Layout:
        [Episode Memory]
        Strategic plan: ...
        Tactical plan: ...
        Subgoals:
          [x] done item
          [ ] pending item
          (none)
        Lessons learned this episode:
          - ...
          (none)

        [Recent Actions]
        turn N: ACTION — outcome
        ...
        (none yet)

        [Current Observation]
        {grid/legend/hud/message rendered}

        [Your turn]  (or [Your turn — FIRST TURN] on turn 0)
    """
    sections: list[str] = []

    # Episode memory
    mem_lines = ["[Episode Memory]"]
    mem_lines.append(f"Strategic plan: {state.strategic_plan or '(not set)'}")
    mem_lines.append(f"Tactical plan: {state.tactical_plan or '(not set)'}")
    mem_lines.append("Subgoals:")
    if state.subgoals:
        for sg in state.subgoals:
            marker = "[x]" if sg.done else "[ ]"
            mem_lines.append(f"  {marker} {sg.text}")
    else:
        mem_lines.append("  (none)")
    mem_lines.append("Lessons learned this episode:")
    if state.lessons:
        for lesson in state.lessons:
            mem_lines.append(f"  - {lesson}")
    else:
        mem_lines.append("  (none)")
    sections.append("\n".join(mem_lines))

    # Recent actions
    recent_lines = ["[Recent Actions]"]
    if state.recent_actions:
        for turn_n, action_name, outcome in state.recent_actions:
            recent_lines.append(f"turn {turn_n}: {action_name} — {outcome}")
    else:
        recent_lines.append("(none yet)")
    sections.append("\n".join(recent_lines))

    # Current observation
    sections.append(f"[Current Observation]\n{current_obs.render()}")

    # Your turn marker
    if turn_index == 0:
        sections.append(
            "[Your turn — FIRST TURN]\n"
            "This is the first turn of the episode. You have no prior plan, "
            "subgoals, or lessons. In your JSON response, produce an initial "
            "strategic plan, an initial list of subgoals, and your first action. "
            "Tactical plan and lessons are optional on turn 0."
        )
    else:
        sections.append("[Your turn]\nOutput the JSON now.")

    return "\n\n".join(sections)
