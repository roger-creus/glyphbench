"""MiniHack suite. Importing this module registers all MiniHack envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/minihack-room-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoom5x5Env",
    max_episode_steps=None,
)
