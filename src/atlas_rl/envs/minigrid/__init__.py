"""MiniGrid suite. Importing this module registers all MiniGrid envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/minigrid-empty-5x5-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmpty5x5Env",
    max_episode_steps=None,
)
