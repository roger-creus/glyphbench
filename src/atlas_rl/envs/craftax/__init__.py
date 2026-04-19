"""Craftax suite. Importing this module registers all Craftax envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/craftax-classic-v0",
    "atlas_rl.envs.craftax.classic:CraftaxClassicEnv",
    max_episode_steps=None,
)
