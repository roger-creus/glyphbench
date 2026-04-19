"""Atari suite. Importing this module registers all Atari envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/atari-pong-v0",
    "atlas_rl.envs.atari.pong:PongEnv",
    max_episode_steps=None,
)
