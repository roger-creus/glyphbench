"""Test-fixture dummy env. Importing this module registers the env with gym."""

from rl_world_ascii.core.registry import register_env

register_env(
    "rl_world_ascii/__dummy-v0",
    "rl_world_ascii.envs.dummy.env:DummyEnv",
    max_episode_steps=None,
)
