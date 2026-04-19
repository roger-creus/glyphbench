"""Shared base for Atari envs.

Each Atari env has its own action spec (unlike MiniGrid/MiniHack which share).
This module provides common Atari rendering constants.
"""

from __future__ import annotations

# Atari-wide rendering constants
COURT_BORDER_H = "-"
COURT_BORDER_V = "|"
COURT_CORNER = "+"
EMPTY_CELL = " "
