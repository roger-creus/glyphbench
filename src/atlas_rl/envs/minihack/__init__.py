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

# --- River ---
register_env(
    "atlas_rl/minihack-river-v0",
    "atlas_rl.envs.minihack.river:MiniHackRiverEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-river-narrow-v0",
    "atlas_rl.envs.minihack.river:MiniHackRiverNarrowEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-river-monster-v0",
    "atlas_rl.envs.minihack.river:MiniHackRiverMonsterEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-river-lava-v0",
    "atlas_rl.envs.minihack.river:MiniHackRiverLavaEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-river-monsterlava-v0",
    "atlas_rl.envs.minihack.river:MiniHackRiverMonsterLavaEnv",
    max_episode_steps=None,
)

# --- HideNSeek ---
register_env(
    "atlas_rl/minihack-hidenseek-v0",
    "atlas_rl.envs.minihack.hidenseek:MiniHackHideNSeekEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-hidenseek-mapped-v0",
    "atlas_rl.envs.minihack.hidenseek:MiniHackHideNSeekMappedEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-hidenseek-lava-v0",
    "atlas_rl.envs.minihack.hidenseek:MiniHackHideNSeekLavaEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-hidenseek-big-v0",
    "atlas_rl.envs.minihack.hidenseek:MiniHackHideNSeekBigEnv",
    max_episode_steps=None,
)

# --- CorridorBattle ---
register_env(
    "atlas_rl/minihack-corridorbattle-v0",
    "atlas_rl.envs.minihack.corridorbattle:MiniHackCorridorBattleEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-corridorbattle-dark-v0",
    "atlas_rl.envs.minihack.corridorbattle:MiniHackCorridorBattleDarkEnv",
    max_episode_steps=None,
)

# --- Skill: Eat ---
register_env(
    "atlas_rl/minihack-eat-v0",
    "atlas_rl.envs.minihack.skill_eat:MiniHackEatEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-eat-distract-v0",
    "atlas_rl.envs.minihack.skill_eat:MiniHackEatDistractEnv",
    max_episode_steps=None,
)

# --- Skill: Pray ---
register_env(
    "atlas_rl/minihack-pray-v0",
    "atlas_rl.envs.minihack.skill_pray:MiniHackPrayEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-pray-distract-v0",
    "atlas_rl.envs.minihack.skill_pray:MiniHackPrayDistractEnv",
    max_episode_steps=None,
)

# --- Skill: Sink ---
register_env(
    "atlas_rl/minihack-sink-v0",
    "atlas_rl.envs.minihack.skill_sink:MiniHackSinkEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-sink-distract-v0",
    "atlas_rl.envs.minihack.skill_sink:MiniHackSinkDistractEnv",
    max_episode_steps=None,
)

# --- Skill: Read ---
register_env(
    "atlas_rl/minihack-read-v0",
    "atlas_rl.envs.minihack.skill_read:MiniHackReadEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-read-distract-v0",
    "atlas_rl.envs.minihack.skill_read:MiniHackReadDistractEnv",
    max_episode_steps=None,
)

# --- Skill: Quaff ---
register_env(
    "atlas_rl/minihack-quaff-v0",
    "atlas_rl.envs.minihack.skill_quaff:MiniHackQuaffEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-quaff-distract-v0",
    "atlas_rl.envs.minihack.skill_quaff:MiniHackQuaffDistractEnv",
    max_episode_steps=None,
)

# --- Skill: Wield ---
register_env(
    "atlas_rl/minihack-wield-v0",
    "atlas_rl.envs.minihack.skill_wield:MiniHackWieldEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-wield-distract-v0",
    "atlas_rl.envs.minihack.skill_wield:MiniHackWieldDistractEnv",
    max_episode_steps=None,
)

# --- Skill: Wand of Death (WoD) ---
register_env(
    "atlas_rl/minihack-wod-easy-v0",
    "atlas_rl.envs.minihack.skill_wod:MiniHackWoDEasyEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-wod-medium-v0",
    "atlas_rl.envs.minihack.skill_wod:MiniHackWoDMediumEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-wod-hard-v0",
    "atlas_rl.envs.minihack.skill_wod:MiniHackWoDHardEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-wod-pro-v0",
    "atlas_rl.envs.minihack.skill_wod:MiniHackWoDProEnv",
    max_episode_steps=None,
)

# --- Skill: LavaCross ---
register_env(
    "atlas_rl/minihack-lavacross-full-v0",
    "atlas_rl.envs.minihack.skill_lavacross:MiniHackLavaCrossFullEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-lavacross-levitate-v0",
    "atlas_rl.envs.minihack.skill_lavacross:MiniHackLavaCrossLevitateEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-lavacross-levitate-potion-inv-v0",
    "atlas_rl.envs.minihack.skill_lavacross:MiniHackLavaCrossPotionInvEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-lavacross-levitate-ring-inv-v0",
    "atlas_rl.envs.minihack.skill_lavacross:MiniHackLavaCrossRingInvEnv",
    max_episode_steps=None,
)

# --- Boxoban ---
register_env(
    "atlas_rl/minihack-boxoban-medium-v0",
    "atlas_rl.envs.minihack.boxoban:MiniHackBoxobanMediumEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-boxoban-hard-v0",
    "atlas_rl.envs.minihack.boxoban:MiniHackBoxobanHardEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-boxoban-unfiltered-v0",
    "atlas_rl.envs.minihack.boxoban:MiniHackBoxobanUnfilteredEnv",
    max_episode_steps=None,
)

# --- Memento ---
register_env(
    "atlas_rl/minihack-memento-short-v0",
    "atlas_rl.envs.minihack.memento:MiniHackMementoShortEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-memento-hard-v0",
    "atlas_rl.envs.minihack.memento:MiniHackMementoHardEnv",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-memento-f2-v0",
    "atlas_rl.envs.minihack.memento:MiniHackMementoF2Env",
    max_episode_steps=None,
)
register_env(
    "atlas_rl/minihack-memento-f4-v0",
    "atlas_rl.envs.minihack.memento:MiniHackMementoF4Env",
    max_episode_steps=None,
)
