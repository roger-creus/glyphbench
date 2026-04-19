"""Test-fixture dummy env. Importing this module registers the env with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/__dummy-v0",
    "atlas_rl.envs.dummy.env:DummyEnv",
    max_episode_steps=None,
)
