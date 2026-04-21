"""Test-fixture dummy env. Importing this module registers the env with gym."""

from glyphbench.core.registry import register_env

register_env(
    "glyphbench/__dummy-v0",
    "glyphbench.envs.dummy.env:DummyEnv",
    max_episode_steps=None,
)
