"""Persistent episode state for the harness.

All fields are freely editable by the LLM each turn via its JSON output. The
only persistence guarantees are: (1) fields carry across turns within an
episode, (2) `reset()` clears everything, (3) `recent_actions` is bounded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Subgoal:
    text: str
    done: bool = False


RECENT_ACTIONS_MAX_LEN = 5


@dataclass
class EpisodeState:
    strategic_plan: str = ""
    tactical_plan: str = ""
    subgoals: list[Subgoal] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    recent_actions: deque[tuple[int, str, str]] = field(
        default_factory=lambda: deque(maxlen=RECENT_ACTIONS_MAX_LEN)
    )

    def reset(self) -> None:
        self.strategic_plan = ""
        self.tactical_plan = ""
        self.subgoals.clear()
        self.lessons.clear()
        self.recent_actions.clear()
