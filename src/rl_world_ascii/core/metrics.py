"""Layer-1 per-turn metrics dataclass.

Defined in core so that the harness can import it without depending on the
runner or provider subpackages. Runner (Plan 0.C) is responsible for
aggregating instances of TurnMetrics into parquet files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TurnMetrics:
    """Every turn produces one of these. Schema matches specs §6.5 turns.parquet."""

    turn_index: int
    wall_time_s: float
    reward: float
    terminated: bool
    truncated: bool

    action_index: int
    action_name: str
    action_parse_error: bool
    action_parse_retries: int
    action_fell_back_to_noop: bool

    tokens_in: int
    tokens_out: int
    tokens_reasoning: int
    latency_provider_s: float
    dollar_cost_turn: float

    subgoals_added: int
    subgoals_marked_done: int
    lessons_added: int
    tactical_plan_changed: bool
    strategic_plan_changed: bool

    prompt_char_count: int
    prompt_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
