"""GlyphbenchMultiTurnEnv + load_environment entry point."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import verifiers as vf
from datasets import Dataset

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.registry import REGISTRY, all_glyphbench_env_ids, make_env
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric


DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_N_FRAMES = 4
DEFAULT_NUM_EPISODES = 10
DEFAULT_BASE_SEED = 42


def load_environment(
    env_id: str | list[str] | None = None,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    n_frames: int = DEFAULT_N_FRAMES,
    max_turns: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    seed: int = DEFAULT_BASE_SEED,
    **kwargs: Any,
) -> vf.Environment:
    """Entry point consumed by ``vf-eval`` and ``prime-rl`` orchestrator.

    Args:
        env_id: single id, list of ids, or ``None`` for all registered envs
                (dummy envs excluded when id is ``None``).
        num_episodes: rollouts per env.
        n_frames: history window shown in each user turn.
        max_turns: per-episode turn cap; ``None`` uses each game's own max_turns.
        max_output_tokens: per-turn LLM budget; communicated to the model in
                the system prompt.
        seed: base seed; each episode uses ``seed + episode_idx`` as the
                per-rollout seed.
    """
    _ensure_envs_loaded()
    env_ids = _resolve_env_ids(env_id)
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
    )


def _ensure_envs_loaded() -> None:
    """Force-import all suite __init__.py files so the registry is populated."""
    from glyphbench.envs import _import_all_suites

    _import_all_suites()


def _resolve_env_ids(env_id: str | list[str] | None) -> list[str]:
    if env_id is None:
        return [i for i in all_glyphbench_env_ids() if "__dummy" not in i]
    if isinstance(env_id, str):
        ids = [env_id]
    else:
        ids = list(env_id)
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
        state["episode_return"] = 0.0

        # Populate the prompt now that we have the game instance.
        system_text = build_system_prompt(game, self._max_output_tokens)
        initial_user_text = render_user_turn(
            game,
            frames=state["frames"],
            current_obs=obs_text,
            turn=0,
            max_output_tokens=self._max_output_tokens,
        )
        state["prompt"] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": initial_user_text},
        ]
        return await super().setup_state(state)

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        game: BaseGlyphEnv = state["game"]

        # The last message in `messages` is the assistant's reply.
        last_assistant = ""
        for m in reversed(messages):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "") or ""
                break

        action_idx, action_name, parse_failed = self.parser.parse_action(
            last_assistant, game.action_spec, noop=game.noop_action_name
        )
        if parse_failed:
            state["parse_failures"] += 1

        pre_obs = state["current_obs"]
        obs_text, reward, term, trunc, _info = game.step(action_idx)

        state["frames"].append((pre_obs, action_name, float(reward)))
        state["current_obs"] = obs_text
        state["episode_return"] += float(reward)
        state["terminated"] = bool(term)
        state["truncated"] = bool(trunc)
        state["done"] = bool(term or trunc)

        # Set the per-turn reward on the trajectory step verifiers appended
        # before calling env_response.
        traj = state.get("trajectory", [])
        if traj:
            traj[-1]["reward"] = float(reward)

        next_user = render_user_turn(
            game,
            frames=state["frames"],
            current_obs=obs_text,
            turn=game.turn,
            max_output_tokens=self._max_output_tokens,
        )
        return [{"role": "user", "content": next_user}]

    @vf.stop
    async def is_done(self, state: dict[str, Any]) -> bool:
        return bool(state.get("done", False))

    @vf.cleanup
    async def close_game(self, state: dict[str, Any]) -> None:
        game = state.pop("game", None)
        if game is not None:
            try:
                game.close()
            except (AttributeError, OSError, RuntimeError):
                pass
