"""Core contracts shared by every env and the verifiers integration."""

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import (
    REGISTRY,
    all_glyphbench_env_ids,
    make_env,
    register_env,
)

# TEMPORARY — remove at end of M4 (plan Task 4.7)
BaseAsciiEnv = BaseGlyphEnv

__all__ = [
    "ActionSpec",
    "BaseGlyphEnv",
    "BaseAsciiEnv",  # temporary alias
    "GridObservation",
    "REGISTRY",
    "all_glyphbench_env_ids",
    "make_env",
    "register_env",
]
