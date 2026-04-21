"""Thin wrapper around gym.register that tracks our owned ids."""

from __future__ import annotations

import gymnasium as gym

_REGISTERED: set[str] = set()


def register_env(
    env_id: str,
    entry_point: str,
    *,
    max_episode_steps: int | None = None,
) -> None:
    """Register a gym id. Idempotent across module re-imports.

    Args:
        env_id: The canonical id, e.g. "glyphbench/minigrid-empty-5x5-v0".
        entry_point: "module.path:ClassName" accepted by gymnasium.
        max_episode_steps: Optional gym-level time limit wrapper.
    """
    if env_id in _REGISTERED:
        return
    gym.register(
        id=env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
    )
    _REGISTERED.add(env_id)


def all_glyphbench_env_ids() -> list[str]:
    """Return every registered id as a sorted list."""
    return sorted(_REGISTERED)
