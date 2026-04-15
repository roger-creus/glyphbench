"""HarnessAgent: the per-episode orchestrator.

Wires env + LLM client + persistent state + parser into a single
`run_episode` coroutine. Returns (return, length, per_turn_metrics).
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from rl_world_ascii.core.base_env import BaseAsciiEnv
from rl_world_ascii.core.metrics import TurnMetrics
from rl_world_ascii.harness.mock_client import LLMResponse
from rl_world_ascii.harness.parser import MAX_REPAIR_RETRIES, parse_harness_output
from rl_world_ascii.harness.prompt_builder import build_user_prompt
from rl_world_ascii.harness.schema import HarnessOutput
from rl_world_ascii.harness.state import EpisodeState, Subgoal


@runtime_checkable
class LLMClientLike(Protocol):
    model_id: str
    provider: str

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None,
        seed: int | None,
    ) -> LLMResponse: ...


class HarnessAgent:
    """One agent = one episode. Reuse a new instance per episode."""

    def __init__(
        self,
        env: BaseAsciiEnv,
        client: LLMClientLike,
        *,
        temperature: float,
        max_output_tokens: int,
        model_seed: int | None = None,
    ) -> None:
        self.env = env
        self.client = client
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model_seed = model_seed
        self.state = EpisodeState()

    async def run_episode(
        self, seed: int
    ) -> tuple[float, int, list[TurnMetrics]]:
        self.state.reset()
        _, _ = self.env.reset(seed=seed)
        current_obs = self.env.get_observation()
        system_prompt = self.env.system_prompt()
        episode_return = 0.0
        turn_metrics: list[TurnMetrics] = []
        turn_index = 0

        while True:
            user_prompt = build_user_prompt(self.state, current_obs, turn_index=turn_index)
            metric, action_index, _raw_responses = await self._one_turn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                turn_index=turn_index,
            )

            _, reward, terminated, truncated, info = self.env.step(action_index)
            episode_return += reward
            current_obs = self.env.get_observation()

            # Fill in env-driven fields of the metric
            metric.reward = reward
            metric.terminated = terminated
            metric.truncated = truncated
            turn_metrics.append(metric)

            # Update recent actions history
            outcome = self._describe_outcome(reward, terminated, truncated, info)
            self.state.recent_actions.append((turn_index, metric.action_name, outcome))

            turn_index += 1
            if terminated or truncated:
                break

        return episode_return, turn_index, turn_metrics

    async def _one_turn(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        turn_index: int,
    ) -> tuple[TurnMetrics, int, list[str]]:
        """Run one LLM call with up to MAX_REPAIR_RETRIES repairs. Returns the
        partially-populated TurnMetrics (env-driven fields filled by caller),
        the chosen action index, and the list of raw LLM response texts."""
        noop_action_name = getattr(self.env, "noop_action_name", "NOOP")
        retries_used = 0
        raw_responses: list[str] = []
        result = None
        current_user_prompt = user_prompt

        wall_start = time.perf_counter()
        cumulative_tokens_in = 0
        cumulative_tokens_out = 0
        cumulative_tokens_reasoning = 0
        cumulative_cost = 0.0
        cumulative_latency_provider = 0.0

        while True:
            response = await self.client.complete(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_format=HarnessOutput.model_json_schema(),
                seed=self.model_seed,
            )
            raw_responses.append(response.text)
            cumulative_tokens_in += response.tokens_in
            cumulative_tokens_out += response.tokens_out
            cumulative_tokens_reasoning += response.tokens_reasoning
            cumulative_cost += response.dollar_cost or 0.0
            cumulative_latency_provider += response.latency_s

            result = parse_harness_output(
                response.text, self.env.action_spec, noop_action_name=noop_action_name
            )
            if result.parse_error is None:
                break
            if retries_used >= MAX_REPAIR_RETRIES:
                break
            retries_used += 1
            current_user_prompt = self._build_repair_prompt(user_prompt, result.parse_error)

        wall_elapsed = time.perf_counter() - wall_start

        # Apply state updates if parse succeeded
        if result is not None and result.parsed is not None:
            self._apply_state_updates(result.parsed)

        assert result is not None  # loop always sets it
        metric = TurnMetrics(
            turn_index=turn_index,
            wall_time_s=wall_elapsed,
            reward=0.0,  # filled by caller
            terminated=False,  # filled by caller
            truncated=False,  # filled by caller
            action_index=result.action_index,
            action_name=result.action_name,
            action_parse_error=result.parse_error is not None,
            action_parse_retries=retries_used,
            action_fell_back_to_noop=result.fell_back_to_noop,
            tokens_in=cumulative_tokens_in,
            tokens_out=cumulative_tokens_out,
            tokens_reasoning=cumulative_tokens_reasoning,
            latency_provider_s=cumulative_latency_provider,
            dollar_cost_turn=cumulative_cost,
            subgoals_added=len(result.parsed.subgoals_update.add) if result.parsed else 0,
            subgoals_marked_done=len(result.parsed.subgoals_update.mark_done) if result.parsed else 0,
            lessons_added=len(result.parsed.lessons_to_add) if result.parsed else 0,
            tactical_plan_changed=bool(result.parsed.tactical_plan) if result.parsed else False,
            strategic_plan_changed=(result.parsed.strategic_plan_update is not None) if result.parsed else False,
            prompt_char_count=len(system_prompt) + len(user_prompt),
            prompt_token_count=0,  # provider will fill in real value in Plan 0.C if possible
        )
        return metric, metric.action_index, raw_responses

    def _apply_state_updates(self, parsed: HarnessOutput) -> None:
        if parsed.strategic_plan_update is not None:
            self.state.strategic_plan = parsed.strategic_plan_update
        if parsed.tactical_plan:
            self.state.tactical_plan = parsed.tactical_plan
        for text in parsed.subgoals_update.add:
            self.state.subgoals.append(Subgoal(text=text, done=False))
        for idx in parsed.subgoals_update.mark_done:
            if 0 <= idx < len(self.state.subgoals):
                self.state.subgoals[idx].done = True
        for lesson in parsed.lessons_to_add:
            self.state.lessons.append(lesson)

    def _build_repair_prompt(self, original_user_prompt: str, error: str) -> str:
        return (
            "Your previous response could not be parsed.\n"
            f"Error: {error}\n\n"
            "Re-emit ONLY the JSON object described in the system prompt. "
            "Do not include any prose, code fences, or explanation outside the JSON. "
            "The single required field is `action`; it must be one of the action "
            "names listed in the system prompt."
        )

    @staticmethod
    def _describe_outcome(
        reward: float, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> str:
        parts: list[str] = []
        if reward != 0.0:
            parts.append(f"reward={reward:+g}")
        if terminated:
            parts.append("terminated")
        if truncated:
            parts.append("truncated")
        if not parts:
            return "no change"
        return ", ".join(parts)
