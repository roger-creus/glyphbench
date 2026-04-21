"""Uniform-random-action agent with the same interface as HarnessAgent."""

from __future__ import annotations

import time

import numpy as np

from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.core.metrics import TurnMetrics


class RandomAgent:
    def __init__(self, *, env: BaseAsciiEnv, seed: int) -> None:
        self.env = env
        self._rng = np.random.default_rng(seed)

    async def run_episode(
        self, seed: int
    ) -> tuple[float, int, list[TurnMetrics]]:
        self.env.reset(seed=seed)
        episode_return = 0.0
        turn_metrics: list[TurnMetrics] = []
        turn_index = 0

        while True:
            wall_start = time.perf_counter()
            action_index = int(self._rng.integers(0, self.env.action_spec.n))
            action_name = self.env.action_spec.names[action_index]
            _, reward, terminated, truncated, _ = self.env.step(action_index)
            episode_return += reward
            wall_elapsed = time.perf_counter() - wall_start

            turn_metrics.append(
                TurnMetrics(
                    turn_index=turn_index,
                    wall_time_s=wall_elapsed,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    action_index=action_index,
                    action_name=action_name,
                    action_parse_error=False,
                    action_parse_retries=0,
                    action_fell_back_to_noop=False,
                    tokens_in=0,
                    tokens_out=0,
                    tokens_reasoning=0,
                    latency_provider_s=0.0,
                    dollar_cost_turn=0.0,
                    subgoals_added=0,
                    subgoals_marked_done=0,
                    lessons_added=0,
                    tactical_plan_changed=False,
                    strategic_plan_changed=False,
                    prompt_char_count=0,
                    prompt_token_count=0,
                )
            )
            turn_index += 1
            if terminated or truncated:
                break

        return episode_return, turn_index, turn_metrics
