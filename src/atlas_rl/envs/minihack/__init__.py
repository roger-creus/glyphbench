"""MiniHack suite. Importing this module registers all MiniHack envs with gym."""

from atlas_rl.core.registry import register_env

# Room variants
register_env(
    "atlas_rl/minihack-room-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoom5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-15x15-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoom15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-random-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomRandom5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-random-15x15-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomRandom15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-dark-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomDark5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-dark-15x15-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomDark15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-monster-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomMonster5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-monster-15x15-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomMonster15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-trap-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomTrap5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-trap-15x15-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomTrap15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-ultimate-5x5-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomUltimate5x5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-room-ultimate-15x15-v0",
    "atlas_rl.envs.minihack.room:MiniHackRoomUltimate15x15Env",
    max_episode_steps=None,
)

# --- Corridor ---
register_env(
    "atlas_rl/minihack-corridor-r2-v0",
    "atlas_rl.envs.minihack.corridor:MiniHackCorridorR2Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-corridor-r3-v0",
    "atlas_rl.envs.minihack.corridor:MiniHackCorridorR3Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-corridor-r5-v0",
    "atlas_rl.envs.minihack.corridor:MiniHackCorridorR5Env",
    max_episode_steps=None,
)

# --- KeyRoom ---
register_env(
    "atlas_rl/minihack-keyroom-s5-v0",
    "atlas_rl.envs.minihack.keyroom:MiniHackKeyRoomS5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-keyroom-s15-v0",
    "atlas_rl.envs.minihack.keyroom:MiniHackKeyRoomS15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-keyroom-dark-s5-v0",
    "atlas_rl.envs.minihack.keyroom:MiniHackKeyRoomDarkS5Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-keyroom-dark-s15-v0",
    "atlas_rl.envs.minihack.keyroom:MiniHackKeyRoomDarkS15Env",
    max_episode_steps=None,
)

# --- MazeWalk (dark) ---
register_env(
    "atlas_rl/minihack-mazewalk-9x9-v0",
    "atlas_rl.envs.minihack.mazewalk:MiniHackMazeWalk9x9Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-mazewalk-15x15-v0",
    "atlas_rl.envs.minihack.mazewalk:MiniHackMazeWalk15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-mazewalk-45x19-v0",
    "atlas_rl.envs.minihack.mazewalk:MiniHackMazeWalk45x19Env",
    max_episode_steps=None,
)

# --- MazeWalk Mapped (fully visible) ---
register_env(
    "atlas_rl/minihack-mazewalk-mapped-9x9-v0",
    "atlas_rl.envs.minihack.mazewalk:MiniHackMazeWalkMapped9x9Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-mazewalk-mapped-15x15-v0",
    "atlas_rl.envs.minihack.mazewalk:MiniHackMazeWalkMapped15x15Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-mazewalk-mapped-45x19-v0",
    "atlas_rl.envs.minihack.mazewalk:MiniHackMazeWalkMapped45x19Env",
    max_episode_steps=None,
)
