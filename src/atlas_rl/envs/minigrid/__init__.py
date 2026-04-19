"""MiniGrid suite. Importing this module registers all MiniGrid envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/minigrid-empty-5x5-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmpty5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minigrid-empty-6x6-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmpty6x6Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minigrid-empty-8x8-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmpty8x8Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minigrid-empty-16x16-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmpty16x16Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minigrid-empty-random-5x5-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmptyRandom5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minigrid-empty-random-6x6-v0",
    "atlas_rl.envs.minigrid.empty:MiniGridEmptyRandom6x6Env",
    max_episode_steps=None,
)
