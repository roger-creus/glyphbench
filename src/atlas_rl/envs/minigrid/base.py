"""Shared base class for all MiniGrid envs.

MiniGrid envs share a common 7-action space and wall-rendering conventions.
Individual envs override mechanics (grid contents, goal conditions, etc.).
"""

from __future__ import annotations

from atlas_rl.core.action import ActionSpec

# Shared MiniGrid action spec (7 actions, matching MiniGrid's native set)
MINIGRID_ACTION_SPEC = ActionSpec(
    names=(
        "TURN_LEFT",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "PICKUP",
        "DROP",
        "TOGGLE",
        "DONE",
    ),
    descriptions=(
        "rotate 90 degrees counter-clockwise",
        "rotate 90 degrees clockwise",
        "move one cell in the direction you are facing",
        "pick up an object in the cell ahead (no-op in Empty)",
        "drop the carried object in the cell ahead (no-op in Empty)",
        "toggle/activate the object in the cell ahead (no-op in Empty)",
        "declare the task is done (no-op in Empty)",
    ),
)

# Direction constants: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3

# Direction vectors: (dx, dy) for each direction
DIR_TO_VEC: dict[int, tuple[int, int]] = {
    DIR_RIGHT: (1, 0),
    DIR_DOWN: (0, 1),
    DIR_LEFT: (-1, 0),
    DIR_UP: (0, -1),
}

# Agent glyphs per direction
DIR_TO_CHAR: dict[int, str] = {
    DIR_RIGHT: ">",
    DIR_DOWN: "v",
    DIR_LEFT: "<",
    DIR_UP: "^",
}
