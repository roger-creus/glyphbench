"""Rubric: sums per-step rewards across the rollout + tracks monitor metrics.

The primary reward is ``episodic_return`` — weight 1.0, summed across every
step of the rollout. All other functions are ``weight=0`` metrics for
observability (parse-failure rate, episode length, terminated/truncated flags,
XML format compliance).
"""

from __future__ import annotations

from typing import Any

import verifiers as vf


class EpisodicReturnRubric(vf.Rubric):
    def __init__(self, parser: vf.Parser | None = None, **kwargs: Any) -> None:
        super().__init__(parser=parser, **kwargs)
        self.add_reward_func(self.episodic_return, weight=1.0)
        self.add_metric(self.episode_length)
        self.add_metric(self.parse_failure_rate)
        self.add_metric(self.terminated_flag)
        self.add_metric(self.truncated_flag)
        if parser is not None:
            try:
                fmt = parser.get_format_reward_func()
                fmt.__name__ = "xml_format_reward"
                self.add_metric(fmt)
            except AttributeError:
                pass  # parser doesn't expose a format reward fn — skip

    async def episodic_return(self, state: dict[str, Any]) -> float:
        return float(state.get("episode_return", 0.0))

    async def episode_length(self, state: dict[str, Any]) -> float:
        # Use num_turns (env-action turns) rather than len(trajectory):
        # in memory mode we emit two trajectory steps (action + memory)
        # per env turn, so trajectory length would double-count.
        n = state.get("num_turns")
        if n is None:
            n = len(state.get("trajectory", []))
        return float(n)

    async def parse_failure_rate(self, state: dict[str, Any]) -> float:
        n = state.get("num_turns")
        if n is None:
            n = len(state.get("trajectory", []))
        if n == 0:
            return 0.0
        return float(state.get("parse_failures", 0)) / float(n)

    async def terminated_flag(self, state: dict[str, Any]) -> float:
        return 1.0 if state.get("terminated") else 0.0

    async def truncated_flag(self, state: dict[str, Any]) -> float:
        return 1.0 if state.get("truncated") else 0.0
