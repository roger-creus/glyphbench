"""Plain-Python class-object registry for glyphbench environments."""

from __future__ import annotations

from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv

REGISTRY: dict[str, type[BaseGlyphEnv]] = {}


def register_env(env_id: str, cls: type[BaseGlyphEnv]) -> None:
    """Register a class under an env id.

    Idempotent for the same (id, class) pair; raises ``ValueError`` on
    conflicting registrations and ``TypeError`` if ``cls`` is not a
    ``BaseGlyphEnv`` subclass.
    """
    if not isinstance(cls, type) or not issubclass(cls, BaseGlyphEnv):
        raise TypeError(
            f"register_env expected a BaseGlyphEnv subclass, got {cls!r}"
        )
    existing = REGISTRY.get(env_id)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"env_id {env_id!r} already registered to {existing.__name__}; "
            f"refusing to overwrite with {cls.__name__}"
        )
    REGISTRY[env_id] = cls


def make_env(env_id: str, **kwargs: Any) -> BaseGlyphEnv:
    """Instantiate the class registered under ``env_id``.

    Extra kwargs are forwarded to the class constructor.
    """
    cls = REGISTRY.get(env_id)
    if cls is None:
        raise KeyError(
            f"unknown env_id {env_id!r}; known ids: {sorted(REGISTRY)[:5]}…"
        )
    return cls(**kwargs)


def all_glyphbench_env_ids() -> list[str]:
    """Return every registered id as a sorted list."""
    return sorted(REGISTRY)
