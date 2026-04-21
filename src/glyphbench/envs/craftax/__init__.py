"""Craftax suite. Importing this module registers all Craftax envs with gym."""

from glyphbench.core.registry import register_env

register_env(
    "glyphbench/craftax-classic-v0",
    "glyphbench.envs.craftax.classic:CraftaxClassicEnv",
    max_episode_steps=None,
)

register_env(
    "glyphbench/craftax-v0",
    "glyphbench.envs.craftax.full:CraftaxFullEnv",
    max_episode_steps=None,
)

# -- Craftax sub-task environments --
register_env(
    "glyphbench/craftax-choptrees-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxChopTreesEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-minestone-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxMineStoneEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-gatherresources-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxGatherResourcesEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-craftpickaxe-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxCraftPickaxeEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-craftsword-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxCraftSwordEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-craftchain-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxCraftChainEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-fightzombie-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxFightZombieEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-survivehorde-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxSurviveHordeEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-dungeonexplore-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxDungeonExploreEnv",
    max_episode_steps=None,
)
register_env(
    "glyphbench/craftax-dungeonclear-v0",
    "glyphbench.envs.craftax.subtasks:CraftaxDungeonClearEnv",
    max_episode_steps=None,
)

# -- Extended Craftax sub-task environments (23 focused tasks) --
import glyphbench.envs.craftax.subtasks_extended  # noqa: F401, E402
