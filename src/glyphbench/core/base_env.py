"""Base class for every game in glyphbench.

Plain Python class — no framework inheritance. Subclasses implement the five
abstract methods; the public reset/step surface is fixed here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation


class BaseGlyphEnv(ABC):
    """Base class for every env in glyphbench.

    Subclasses MUST:
      - set `action_spec: ActionSpec` as a class or instance attribute
      - implement `_reset(seed)` returning the initial GridObservation
      - implement `_step(action_index)` returning (obs, reward, terminated, truncated, info)
      - implement `_render_current_observation()` returning current state as GridObservation
      - implement `system_prompt()` returning the per-game system prompt
      - implement `env_id()` returning the canonical env id string
    """

    action_spec: ActionSpec
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 500) -> None:
        self.max_turns = max_turns
        self._turn: int = 0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> tuple[str, dict[str, Any]]:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self._rng = np.random.default_rng(int(seed))
        self._turn = 0
        obs = self._reset(int(seed))
        info: dict[str, Any] = {
            "turn": 0,
            "env_id": self.env_id(),
            "seed": int(seed),
        }
        return obs.render(), info

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if not 0 <= int(action) < self.action_spec.n:
            raise ValueError(
                f"action {action} out of range [0, {self.action_spec.n})"
            )
        self._turn += 1
        obs, reward, terminated, truncated, info = self._step(int(action))
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
    def system_prompt(self) -> str:
        """Per-game rules / goal description for the system prompt.

        Should describe the objective, reward structure, and any game-specific
        mechanics. Should NOT enumerate the action list — the verifiers
        integration appends ``action_spec.render_for_prompt()`` separately
        (see ``build_system_prompt`` in ``verifiers_integration.prompting``),
        so including it here would duplicate tokens.
        """
        ...

    @abstractmethod
    def env_id(self) -> str: ...

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("call reset() before accessing rng")
        return self._rng

    @property
    def turn(self) -> int:
        """Number of steps taken since the last reset()."""
        return self._turn

    def get_observation(self) -> GridObservation:
        """Return the current observation without stepping. Useful for initial
        prompt construction at turn 0."""
        return self._render_current_observation()

    def close(self) -> None:
        """Optional cleanup hook. Default: no-op."""
        return None
