"""Procgen suite. Importing this module registers all Procgen envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/procgen-coinrun-v0",
    "atlas_rl.envs.procgen.coinrun:CoinRunEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/procgen-maze-v0",
    "atlas_rl.envs.procgen.maze:MazeEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/procgen-heist-v0",
    "atlas_rl.envs.procgen.heist:HeistEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/procgen-leaper-v0",
    "atlas_rl.envs.procgen.leaper:LeaperEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/procgen-chaser-v0",
    "atlas_rl.envs.procgen.chaser:ChaserEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/procgen-bigfish-v0",
    "atlas_rl.envs.procgen.bigfish:BigFishEnv",
    max_episode_steps=None,
)
