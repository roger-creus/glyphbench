"""The abstract base class every env extends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np

from rl_world_ascii.core.action import ActionSpec
from rl_world_ascii.core.observation import GridObservation


class BaseAsciiEnv(gym.Env[str, int], ABC):
    """Base class for every env in rl_world_ascii.

    Subclasses MUST:
      - set `action_spec: ActionSpec` as a class or instance attribute
      - implement `_reset(seed)` returning the initial GridObservation
      - implement `_step(action_index)` returning (obs, reward, terminated, truncated, info)
      - implement `_render_current_observation()` returning current state as GridObservation
      - implement `system_prompt()` returning the per-game system prompt
      - implement `env_id()` returning the gym env id string
    """

    metadata = {"render_modes": ["ansi"]}

    action_spec: ActionSpec  # class attribute set by subclass
    noop_action_name: str = "NOOP"  # subclasses override to match their action set

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__()
        self.max_turns = max_turns
        self.action_space = gym.spaces.Discrete(self.action_spec.n)
        self.observation_space = gym.spaces.Text(max_length=1 << 16)
        self._turn: int = 0
        self._rng: np.random.Generator | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if seed is None:
            raise ValueError("rl_world_ascii envs require an explicit integer seed on reset()")
        self._rng = np.random.default_rng(seed)
        self._turn = 0
        obs = self._reset(seed)
        info: dict[str, Any] = {
            "turn": 0,
            "env_id": self.env_id(),
            "seed": seed,
        }
        return obs.render(), info

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if not 0 <= int(action) < self.action_spec.n:
            raise ValueError(
                f"action {action} out of range [0, {self.action_spec.n})"
            )
        obs, reward, terminated, truncated, info = self._step(int(action))
        self._turn += 1
        if self._turn >= self.max_turns and not (terminated or truncated):
            truncated = True
            info["truncation_reason"] = "max_turns"
        info["turn"] = self._turn
        info["env_id"] = self.env_id()
        return obs.render(), float(reward), terminated, truncated, info

    @abstractmethod
    def _reset(self, seed: int) -> GridObservation: ...

    @abstractmethod
    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]: ...

    @abstractmethod
    def _render_current_observation(self) -> GridObservation: ...

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def env_id(self) -> str: ...

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("call reset() before accessing rng")
        return self._rng

    def get_observation(self) -> GridObservation:
        """Return the current observation without stepping. Used by the harness
        for the initial prompt at turn 0."""
        return self._render_current_observation()
