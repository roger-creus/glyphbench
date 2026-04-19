"""Shared base class for all MiniHack envs.

MiniHack envs share a common 15-action space (NetHack-inspired) and
dungeon-style ASCII rendering conventions.
"""

from __future__ import annotations

from atlas_rl.core.action import ActionSpec

# Shared MiniHack action spec (15 actions)
MINIHACK_ACTION_SPEC = ActionSpec(
    names=(
        "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
        "MOVE_NE", "MOVE_NW", "MOVE_SE", "MOVE_SW",
        "WAIT", "SEARCH", "LOOK",
        "PICKUP", "APPLY", "INVENTORY", "ESCAPE",
    ),
    descriptions=(
        "move one cell north (up)",
        "move one cell south (down)",
        "move one cell east (right)",
        "move one cell west (left)",
        "move one cell northeast (up-right)",
        "move one cell northwest (up-left)",
        "move one cell southeast (down-right)",
        "move one cell southwest (down-left)",
        "wait one turn (no-op)",
        "search the area around you (no-op in Room)",
        "look around (no-op in Room)",
        "pick up an item at your feet (no-op in Room)",
        "apply/use an item (no-op in Room)",
        "check your inventory (no-op in Room)",
        "escape/cancel (no-op in Room)",
    ),
)

# Direction vectors for 8-directional movement
MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "MOVE_N": (0, -1),
    "MOVE_S": (0, 1),
    "MOVE_E": (1, 0),
    "MOVE_W": (-1, 0),
    "MOVE_NE": (1, -1),
    "MOVE_NW": (-1, -1),
    "MOVE_SE": (1, 1),
    "MOVE_SW": (-1, 1),
}
