"""Core contracts shared by every env and the harness.

Public types:
    GridObservation — the frozen observation dataclass every env returns.
    ActionSpec — per-env action vocabulary.
    BaseAsciiEnv — abstract base class for all envs.

Functions:
    register_env — register a gym id pointing at our entry points.
    all_glyphbench_env_ids — list all registered ids.
"""

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import all_glyphbench_env_ids, register_env

__all__ = [
    "ActionSpec",
    "BaseAsciiEnv",
    "GridObservation",
    "all_glyphbench_env_ids",
    "register_env",
]
