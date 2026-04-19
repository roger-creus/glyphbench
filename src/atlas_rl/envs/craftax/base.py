"""Shared base for Craftax envs.

Provides constants for biome types, resource types, and the shared action spec.
"""

from __future__ import annotations

from atlas_rl.core.action import ActionSpec

# Shared Craftax action spec (19 actions)
CRAFTAX_ACTION_SPEC = ActionSpec(
    names=(
        "NOOP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
        "DO", "SLEEP",
        "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT",
        "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE",
        "MAKE_IRON_PICKAXE",
        "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD",
        "MAKE_IRON_SWORD",
        "EAT_PLANT", "DRINK_WATER",
    ),
    descriptions=(
        "do nothing this turn",
        "move one cell left",
        "move one cell right",
        "move one cell up",
        "move one cell down",
        "interact with the cell you face (chop tree, mine stone/ore, attack mob)",
        "sleep to restore energy (skips 50 day/night steps)",
        "place a stone block in the cell you face (costs 1 stone)",
        "place a crafting table in the cell you face (costs 2 wood)",
        "place a furnace in the cell you face (costs 4 stone)",
        "place a sapling in the cell you face (costs 1 sapling)",
        "craft a wood pickaxe (costs 1 wood, requires adjacent table)",
        "craft a stone pickaxe (costs 1 wood + 1 stone, requires adjacent table+furnace)",
        "craft an iron pickaxe (costs 1 wood + 1 iron, requires adjacent table+furnace)",
        "craft a wood sword (costs 1 wood, requires adjacent table)",
        "craft a stone sword (costs 1 wood + 1 stone, requires adjacent table+furnace)",
        "craft an iron sword (costs 1 wood + 1 iron, requires adjacent table+furnace)",
        "eat a ripe plant you face to restore food",
        "drink water you face to restore thirst",
    ),
)

# Terrain/biome tile characters
TILE_GRASS = "."
TILE_TREE = "T"
TILE_STONE = "S"
TILE_COAL = "C"
TILE_IRON = "I"
TILE_DIAMOND = "D"
TILE_WATER = "~"
TILE_LAVA = "L"
TILE_SAND = "s"
TILE_AGENT = "@"
TILE_TABLE = "t"
TILE_FURNACE = "f"
TILE_PLACED_STONE = "="
TILE_PLANT = "+"

# Mob tile characters
TILE_ZOMBIE = "z"
TILE_SKELETON = "k"
TILE_COW = "c"

# Plant tile characters
TILE_SAPLING = ";"
TILE_RIPE_PLANT = "*"

# Stage 0 achievements (first 8, kept for backward compatibility)
STAGE0_ACHIEVEMENTS = (
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "place_furnace",
    "make_stone_pickaxe",
    "collect_iron",
    "collect_coal",
)

# All 22 Craftax Classic achievements
ALL_CLASSIC_ACHIEVEMENTS = (
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "place_furnace",
    "make_stone_pickaxe",
    "collect_iron",
    "collect_coal",
    "place_stone",
    "collect_drink",
    "collect_sapling",
    "place_plant",
    "eat_plant",
    "defeat_zombie",
    "defeat_skeleton",
    "wake_up",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_wood_sword",
    "make_stone_sword",
    "eat_cow",
)

# Visible window dimensions
VIEW_WIDTH = 9
VIEW_HEIGHT = 7
