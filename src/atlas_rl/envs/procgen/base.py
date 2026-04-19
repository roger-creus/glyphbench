"""Shared base for Procgen envs.

Procgen envs share procedural level generation patterns, integer-cell physics,
and partial-observation windowing.
"""

from __future__ import annotations

# Jump arc definition: sequence of dy values per step while jumping.
# Positive dy = move down (screen coords); negative dy = move up.
# This arc rises for 3 steps, plateaus for 1 step, then falls for 3 steps.
JUMP_ARC_DY: tuple[int, ...] = (-1, -1, -1, 0, 1, 1, 1)

# Partial observation window size
VIEW_WIDTH = 20
VIEW_HEIGHT = 12
