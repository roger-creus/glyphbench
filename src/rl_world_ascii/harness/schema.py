"""Pydantic models for the LLM's per-turn output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubgoalsUpdate(BaseModel):
    add: list[str] = Field(default_factory=list, description="New subgoal texts to append")
    mark_done: list[int] = Field(
        default_factory=list,
        description="Indices into the current subgoals list to mark as done",
    )


class HarnessOutput(BaseModel):
    thinking: str = Field(
        default="",
        description="Free-form chain of thought. Discarded after this turn — "
        "never persisted, never re-shown to you in later turns.",
    )
    strategic_plan_update: str | None = Field(
        default=None,
        description="null means keep the prior strategic plan unchanged; "
        "a string (including empty) replaces it.",
    )
    tactical_plan: str = Field(
        default="",
        description="The new tactical plan for this turn; replaces the prior one.",
    )
    subgoals_update: SubgoalsUpdate = Field(default_factory=SubgoalsUpdate)
    lessons_to_add: list[str] = Field(
        default_factory=list,
        description="New lessons learned this turn; appended to the running list.",
    )
    action: str = Field(
        description="The single action name (SHOUTY_SNAKE_CASE) you choose this turn.",
    )
