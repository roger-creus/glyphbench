"""GlyphbenchMultiTurnEnv + load_environment entry point."""

from __future__ import annotations

import json
from collections import deque
from contextlib import suppress
from typing import Any

import verifiers as vf
from datasets import Dataset
from verifiers.types import Response, TrajectoryStep
from verifiers.utils.response_utils import parse_response_message, parse_response_tokens

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.registry import REGISTRY, all_glyphbench_env_ids, make_env
from glyphbench.verifiers_integration.memory import (
    build_memory_update_user,
    extract_memory_update,
    memory_sampling_args,
)
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

DEFAULT_MAX_OUTPUT_TOKENS = 512
# Stateless per turn by default — every observation must be readable off
# the current grid alone. Frame stacking is opt-in via load_environment.
DEFAULT_N_FRAMES = 0
DEFAULT_NUM_EPISODES = 5
DEFAULT_BASE_SEED = 42


def load_environment(
    task_id: str | list[str] | None = None,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    n_frames: int = DEFAULT_N_FRAMES,
    max_turns: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    seed: int = DEFAULT_BASE_SEED,
    use_memory: bool = False,
    memory_update_max_tokens: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Entry point consumed by ``prime eval run`` and ``prime-rl`` orchestrator.

    Args:
        task_id: single glyphbench env id (e.g. ``"glyphbench/__dummy-v0"``),
                a list of ids, or ``None`` for all registered envs (dummy envs
                excluded when id is ``None``). Named ``task_id`` (not
                ``env_id``) because verifiers reserves ``env_id`` for the
                package name passed via ``vf.load_environment``.
        num_episodes: rollouts per env.
        n_frames: history window shown in each user turn.
        max_turns: per-episode turn cap; ``None`` uses each game's own max_turns.
        max_output_tokens: per-turn LLM budget; communicated to the model in
                the system prompt.
        seed: base seed; each episode uses ``seed + episode_idx`` as the
                per-rollout seed.
        use_memory: when true, each environment step uses an action generation
                followed by a memory-update generation, stored as one trainable
                trajectory segment.
        memory_update_max_tokens: optional generation limit for the memory
                update call. ``None`` reuses the action sampling limit.

    ``**kwargs`` is intentional: verifiers' generic loader injects
    ``env_id="<package-name>"``. We absorb it (and reject anything else) so
    callers can either go through ``vf.load_environment`` or call
    ``glyphbench.load_environment`` directly.
    """
    # verifiers injects env_id=<package name>; ignore that one specific kwarg.
    kwargs.pop("env_id", None)
    if kwargs:
        raise TypeError(
            f"load_environment() got unexpected keyword arguments: "
            f"{sorted(kwargs)!r}. Did you mean 'task_id' for the glyphbench "
            f"env id?"
        )
    _ensure_envs_loaded()
    env_ids = _resolve_env_ids(task_id)
    dataset = _build_dataset(env_ids, num_episodes, seed)

    parser = GlyphbenchXMLParser()
    rubric = EpisodicReturnRubric(parser=parser)

    return GlyphbenchMultiTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        n_frames=n_frames,
        max_turns_override=max_turns,
        max_output_tokens=max_output_tokens,
        use_memory=use_memory,
        memory_update_max_tokens=memory_update_max_tokens,
    )


def _ensure_envs_loaded() -> None:
    """Force-import all suite __init__.py files so the registry is populated."""
    from glyphbench.envs import _import_all_suites

    _import_all_suites()


def _resolve_env_ids(env_id: str | list[str] | None) -> list[str]:
    if env_id is None:
        return [i for i in all_glyphbench_env_ids() if "__dummy" not in i]
    ids = [env_id] if isinstance(env_id, str) else list(env_id)
    missing = [i for i in ids if i not in REGISTRY]
    if missing:
        raise KeyError(
            f"unknown env_id(s): {missing!r}. "
            f"Known ids (sample): {sorted(REGISTRY)[:5]}…"
        )
    return ids


def _build_dataset(env_ids: list[str], num_episodes: int, base_seed: int) -> Dataset:
    rows = []
    for env_id in env_ids:
        for ep in range(num_episodes):
            seed_val = int(base_seed) + ep
            rows.append(
                {
                    "info": json.dumps({"env_id": env_id, "seed": seed_val}),
                    "task": env_id,
                    # Placeholder — filled in setup_state (verifiers allows
                    # dynamic prompt construction via state["prompt"] mutation).
                    "prompt": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": ""},
                    ],
                    "answer": "",
                }
            )
    return Dataset.from_list(rows)


class GlyphbenchMultiTurnEnv(vf.MultiTurnEnv):
    """Verifiers MultiTurnEnv that drives a glyphbench game per rollout."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        rubric: vf.Rubric,
        parser: GlyphbenchXMLParser,
        n_frames: int,
        max_turns_override: int | None,
        max_output_tokens: int,
        use_memory: bool = False,
        memory_update_max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        effective_max_turns = max_turns_override if max_turns_override is not None else -1
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            max_turns=effective_max_turns,
            **kwargs,
        )
        self.n_frames = int(n_frames)
        self._max_turns_override = max_turns_override
        self._max_output_tokens = int(max_output_tokens)
        self._use_memory = bool(use_memory)
        self._memory_update_max_tokens = (
            None
            if memory_update_max_tokens is None
            else int(memory_update_max_tokens)
        )
        self.parser: GlyphbenchXMLParser = parser  # narrow type

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        info_raw = state.get("info", {})
        info = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
        env_id = info["env_id"]
        seed_val = int(info["seed"])

        kw: dict[str, Any] = {}
        if self._max_turns_override is not None:
            kw["max_turns"] = self._max_turns_override

        game = make_env(env_id, **kw)
        obs_text, _ = game.reset(seed_val)

        state["game"] = game
        state["frames"] = deque(maxlen=self.n_frames)
        state["current_obs"] = obs_text
        state["done"] = False
        state["terminated"] = False
        state["truncated"] = False
        state["parse_failures"] = 0
        state["forfeit_count"] = 0
        state["episode_return"] = 0.0
        state["num_turns"] = 0
        state["memory_enabled"] = self._use_memory
        state["memory"] = ""

        # Populate the prompt now that we have the game instance.
        system_text = build_system_prompt(
            game,
            self._max_output_tokens,
            use_memory=self._use_memory,
            memory_update_max_tokens=self._memory_update_max_tokens,
        )
        initial_user_text = self._render_action_user(game, state, turn=0)
        state["prompt"] = [
            vf.SystemMessage(content=system_text),
            vf.UserMessage(content=initial_user_text),
        ]
        await super().setup_state(state)
        return state

    def _render_action_user(
        self, game: BaseGlyphEnv, state: dict[str, Any], *, turn: int
    ) -> str:
        memory = state.get("memory", "") if self._use_memory else None
        return self._render_observation_user(game, state, turn=turn, memory=memory)

    def _render_observation_user(
        self,
        game: BaseGlyphEnv,
        state: dict[str, Any],
        *,
        turn: int,
        memory: str | None,
    ) -> str:
        return render_user_turn(
            game,
            frames=state["frames"],
            current_obs=state["current_obs"],
            turn=turn,
            max_output_tokens=self._max_output_tokens,
            memory=memory,
        )

    def _last_assistant_text(self, messages: list[dict[str, Any]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = m.get("content", "") or ""
                return content if isinstance(content, str) else str(content)
        return ""

    def _apply_action_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        game: BaseGlyphEnv = state["game"]
        last_assistant = self._last_assistant_text(messages)

        action_idx, action_name, parse_failed, parse_failure_reason = (
            self.parser.parse_action(
                last_assistant, game.action_spec, noop=game.noop_action_name
            )
        )

        pre_obs = state["current_obs"]
        if parse_failed:
            state["parse_failures"] += 1
            state["forfeit_count"] = state.get("forfeit_count", 0) + 1
            obs_text, reward, term, trunc, _info = game.forfeit_turn()
            applied_action_name = "FORFEIT"
            applied_action_idx = -1
            forfeit = True
        else:
            obs_text, reward, term, trunc, _info = game.step(action_idx)
            applied_action_name = action_name
            applied_action_idx = action_idx
            forfeit = False

        state["frames"].append((pre_obs, applied_action_name, float(reward)))
        state["current_obs"] = obs_text
        state["episode_return"] += float(reward)
        state["terminated"] = bool(term)
        state["truncated"] = bool(trunc)
        state["done"] = bool(term or trunc)

        return {
            "action_idx": applied_action_idx,
            "action_name": applied_action_name,
            "action_chosen": applied_action_name,
            "parse_failed": parse_failed,
            "parse_failure_reason": parse_failure_reason,
            "forfeit": forfeit,
            "pre_obs": pre_obs,
            "next_obs": obs_text,
            "reward": float(reward),
            "terminated": bool(term),
            "truncated": bool(trunc),
        }

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        game: BaseGlyphEnv = state["game"]

        result = self._apply_action_response(messages, state)

        # Set the per-turn reward on the trajectory step verifiers appended
        # before calling env_response.
        traj = state.get("trajectory", [])
        if traj:
            traj[-1]["reward"] = float(result["reward"])

        next_user = self._render_action_user(game, state, turn=game.turn)
        response_msg = [vf.UserMessage(content=next_user)]
        if state["done"]:
            # Game ended this turn — signal the rollout loop to skip the
            # otherwise-wasted final model call.
            state["final_env_response"] = response_msg
        return response_msg

    async def get_prompt_messages(self, state: dict[str, Any]) -> list[Any]:
        """Stateless prompt: each turn the LLM sees only [system, current_user_obs]."""
        if self._use_memory:
            if len(state["trajectory"]) == 0:
                return state["prompt"]
            game: BaseGlyphEnv = state["game"]
            system_msg = state["prompt"][0]
            next_user = self._render_action_user(game, state, turn=game.turn)
            return [system_msg, vf.UserMessage(content=next_user)]
        if len(state["trajectory"]) == 0:
            return state["prompt"]
        prev = state["trajectory"][-1]
        messages = list(prev["prompt"]) + list(prev["completion"])
        new_user = await self.env_response(messages, state)
        system_msg = prev["prompt"][0]
        return [system_msg] + list(new_user)

    async def add_model_response(
        self,
        state: dict[str, Any],
        prompt_messages: list[Any],
        response: Response,
    ) -> None:
        if not self._use_memory:
            await super().add_model_response(state, prompt_messages, response)
            state["num_turns"] = state.get("num_turns", 0) + 1
            return

        # Memory mode: each environment turn produces TWO trajectory steps so
        # every step's completion is purely assistant tokens (prime-rl's
        # pretokenize_rollout_trajectory rejects mixed-role completions).
        #
        #   action step:  prompt=[system, user_obs_t]
        #                 completion=[assistant_action_t]
        #   memory step:  prompt=[system, user_obs_t, assistant_action_t, memory_user_t]
        #                 completion=[assistant_memory_response_t]
        #
        # Both steps share the same per-turn env reward and (downstream) the
        # same advantage, so the trainer treats memory tokens as on-policy
        # too.
        action_completion = await parse_response_message(response)
        action_tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
        action_is_truncated = response_is_truncated or (
            action_tokens is not None and bool(action_tokens.get("is_truncated"))
        )

        messages_for_action = list(prompt_messages) + list(action_completion)
        action_result = self._apply_action_response(messages_for_action, state)
        state["num_turns"] = state.get("num_turns", 0) + 1
        turn_reward = float(action_result["reward"])

        action_step = TrajectoryStep(
            prompt=prompt_messages,
            completion=action_completion,
            response=response,
            tokens=action_tokens,
            reward=turn_reward,
            advantage=None,
            is_truncated=action_is_truncated,
            trajectory_id=state["trajectory_id"],
            extras={
                "glyphbench_step_role": "action",
                "parse_failed": bool(action_result["parse_failed"]),
                "parse_failure_reason": action_result["parse_failure_reason"],
                "action_chosen": action_result["action_chosen"],
                "forfeit": bool(action_result["forfeit"]),
            },
        )
        state["trajectory"].append(action_step)

        # Lean memory-update prompt: only env feedback + write instruction.
        memory_user = build_memory_update_user(
            reward=turn_reward,
            terminated=bool(action_result["terminated"]),
            truncated=bool(action_result["truncated"]),
        )
        memory_prompt_messages = messages_for_action + [memory_user]
        memory_response = await self.get_model_response(
            state,
            memory_prompt_messages,
            sampling_args=memory_sampling_args(
                state.get("sampling_args"), self._memory_update_max_tokens
            ),
        )
        memory_completion = await parse_response_message(memory_response)
        memory_tokens = await parse_response_tokens(memory_response, self.max_seq_len)
        memory_response_text = self._last_assistant_text(memory_completion)
        extraction = extract_memory_update(memory_response_text)

        # Retain previous memory on parse failure; otherwise apply the new one.
        if not extraction.parse_failed:
            state["memory"] = extraction.memory
        # state["memory"] otherwise stays unchanged.

        memory_response_is_truncated = memory_response.message.is_truncated or False
        memory_is_truncated = memory_response_is_truncated or (
            memory_tokens is not None and bool(memory_tokens.get("is_truncated"))
        )

        memory_step = TrajectoryStep(
            prompt=memory_prompt_messages,
            completion=memory_completion,
            response=memory_response,
            tokens=memory_tokens,
            reward=turn_reward,
            advantage=None,
            is_truncated=bool(memory_is_truncated),
            trajectory_id=state["trajectory_id"],
            extras={
                "glyphbench_step_role": "memory",
                "memory_parse_failed": bool(extraction.parse_failed),
                "stored_memory": state["memory"],
            },
        )
        state["trajectory"].append(memory_step)

        if state["done"]:
            game: BaseGlyphEnv = state["game"]
            final_user = self._render_action_user(game, state, turn=game.turn)
            state["final_env_response"] = [vf.UserMessage(content=final_user)]

    async def render_completion(self, state: dict[str, Any]) -> None:
        """Stitch full rollout from trajectory by walking each step and appending
        only the messages this step contributes that aren't already in the
        running transcript. Handles both stateless action-only steps (whose
        prompts share only the system message with the previous turn) and
        memory steps (whose prompts also include the just-appended action +
        a new memory_update_user).
        """
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return
        parts: list[Any] = []
        running = list(state["prompt"])
        for step in state["trajectory"]:
            step_prompt = list(step["prompt"])
            i = 0
            while (
                i < len(running)
                and i < len(step_prompt)
                and running[i] == step_prompt[i]
            ):
                i += 1
            new_prompt_tail = step_prompt[i:]
            parts.extend(new_prompt_tail)
            parts.extend(list(step["completion"]))
            running = step_prompt + list(step["completion"])
        if state.get("final_env_response"):
            parts.extend(list(state["final_env_response"]))
        state["completion"] = parts

    @vf.stop
    async def is_done(self, state: dict[str, Any]) -> bool:
        return bool(state.get("done", False))

    @vf.cleanup
    async def close_game(self, state: dict[str, Any]) -> None:
        game = state.pop("game", None)
        if game is not None:
            with suppress(AttributeError, OSError, RuntimeError):
                game.close()
