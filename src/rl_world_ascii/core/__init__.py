"""Core contracts shared by every env and the harness.

Public types:
    GridObservation — the frozen observation dataclass every env returns.
    ActionSpec — per-env action vocabulary.
    BaseAsciiEnv — abstract base class for all envs.

Functions:
    register_env — register a gym id pointing at our entry points.
    all_rl_world_ascii_env_ids — list all registered ids.
"""

from rl_world_ascii.core.action import ActionSpec
from rl_world_ascii.core.base_env import BaseAsciiEnv
from rl_world_ascii.core.observation import GridObservation
from rl_world_ascii.core.registry import all_rl_world_ascii_env_ids, register_env

__all__ = [
    "ActionSpec",
    "BaseAsciiEnv",
    "GridObservation",
    "all_rl_world_ascii_env_ids",
    "register_env",
]
