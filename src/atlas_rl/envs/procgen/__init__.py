"""Procgen suite. Importing this module registers all Procgen envs with gym."""

from atlas_rl.core.registry import register_env

register_env(
    "atlas_rl/procgen-coinrun-v0",
    "atlas_rl.envs.procgen.coinrun:CoinRunEnv",
    max_episode_steps=None,
)
