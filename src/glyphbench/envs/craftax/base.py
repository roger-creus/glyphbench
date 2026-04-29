"""Shared base for Craftax envs.

Provides constants for biome types, resource types, and the shared action spec.
"""

from __future__ import annotations

from glyphbench.core.action import ActionSpec

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
        "craft a stone pickaxe (costs 1 wood + 1 stone, requires adjacent table)",
        "craft an iron pickaxe (costs 1 wood + 1 iron, requires adjacent table+furnace)",
        "craft a wood sword (costs 1 wood, requires adjacent table)",
        "craft a stone sword (costs 1 wood + 1 stone, requires adjacent table)",
        "craft an iron sword (costs 1 wood + 1 iron, requires adjacent table+furnace)",
        "eat a ripe plant you face to restore food",
        "drink water you face to restore thirst",
    ),
)

# Full Craftax action spec (35 actions)
CRAFTAX_FULL_ACTION_SPEC = ActionSpec(
    names=(
        "NOOP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
        "DO", "SLEEP",
        "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE",
        "PLACE_PLANT", "PLACE_TORCH",
        "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE",
        "MAKE_IRON_PICKAXE", "MAKE_DIAMOND_PICKAXE",
        "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD",
        "MAKE_IRON_SWORD", "MAKE_DIAMOND_SWORD",
        "MAKE_WOOD_ARMOR", "MAKE_STONE_ARMOR",
        "MAKE_IRON_ARMOR", "MAKE_DIAMOND_ARMOR",
        "CAST_FIREBALL", "CAST_ICEBALL", "CAST_HEAL",
        "DRINK_POTION", "EAT_PLANT", "DRINK_WATER",
        "DESCEND", "ASCEND",
        "ENCHANT_WEAPON", "ENCHANT_ARMOR",
        "MAKE_SPELL_SCROLL",
    ),
    descriptions=(
        "do nothing this turn",
        "move one cell left",
        "move one cell right",
        "move one cell up",
        "move one cell down",
        "interact with the cell you face (chop/mine/attack)",
        "sleep to restore energy (skips 50 day/night steps)",
        "place a stone block (costs 1 stone)",
        "place a crafting table (costs 2 wood)",
        "place a furnace (costs 4 stone)",
        "place a sapling (costs 1 sapling)",
        "place a torch (costs 1 wood + 1 coal, lights dungeons)",
        "craft wood pickaxe (1 wood, table)",
        "craft stone pickaxe (1 wood+1 stone, table)",
        "craft iron pickaxe (1 wood+1 iron, table+furnace)",
        "craft diamond pickaxe (1 wood+1 diamond, table+furnace)",
        "craft wood sword (1 wood, table)",
        "craft stone sword (1 wood+1 stone, table)",
        "craft iron sword (1 wood+1 iron, table+furnace)",
        "craft diamond sword (1 wood+1 diamond, table+furnace)",
        "craft wood armor (2 wood, table)",
        "craft stone armor (2 stone, table)",
        "craft iron armor (2 iron, table+furnace)",
        "craft diamond armor (1 diamond+1 iron, table+furnace)",
        "cast fireball (3 mana, 4 dmg in 2-tile radius)",
        "cast iceball (2 mana, freeze mob in front 5 turns)",
        "cast heal (4 mana, restore 3 HP)",
        "drink a potion from inventory",
        "eat a ripe plant you face to restore food",
        "drink water you face to restore thirst",
        "descend stairs (\u21e3) to next dungeon floor",
        "ascend stairs (\u21e1) to previous floor",
        "enchant weapon (+2 dmg, diamond+coal, table+furnace)",
        "enchant armor (+1 def, diamond+coal, table+furnace)",
        "craft spell scroll (1 wood+1 coal+1 iron, table+furnace)",
    ),
)

# Terrain/biome tile characters (Unicode)
TILE_GRASS = "\u00b7"          # · middle dot
TILE_TREE = "\u2663"           # ♣ club suit
TILE_STONE = "S"
TILE_COAL = "C"
TILE_IRON = "I"
TILE_DIAMOND = "D"
TILE_WATER = "\u2248"          # ≈ almost equal
TILE_LAVA = "\u2668"           # ♨ hot springs
TILE_SAND = "\u2591"           # ░ light shade
TILE_AGENT = "@"
TILE_TABLE = "t"
TILE_FURNACE = "f"
TILE_PLACED_STONE = "="
TILE_PLANT = "+"

# Dungeon-specific tile characters (Unicode)
TILE_STAIRS_DOWN = "\u21e3"    # ⇣ downwards dashed arrow
TILE_STAIRS_UP = "\u21e1"      # ⇡ upwards dashed arrow
TILE_TORCH = "!"
TILE_DUNGEON_WALL = "\u2588"   # █ full block
TILE_DUNGEON_FLOOR = "\u25aa"  # ▪ black small square
TILE_BOSS_DOOR = "B"

# Mob tile characters
TILE_ZOMBIE = "z"
TILE_SKELETON = "k"
TILE_COW = "c"

# Full-version mob tiles
TILE_SKELETON_ARCHER = "a"
TILE_SPIDER = "x"
TILE_BAT = "b"
TILE_BOSS = "W"

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

# All 80 Craftax Full achievements
ALL_FULL_ACHIEVEMENTS = (
    # -- Classic 22 --
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
    # -- Diamond tier (3) --
    "make_diamond_pickaxe",
    "make_diamond_sword",
    # -- Armor (4) --
    "make_wood_armor",
    "make_stone_armor",
    "make_iron_armor",
    "make_diamond_armor",
    # -- Torch & scrolls (2) --
    "place_torch",
    "make_spell_scroll",
    # -- Magic (3) --
    "cast_fireball",
    "cast_iceball",
    "cast_heal",
    # -- Potions (3) --
    "drink_health_potion",
    "drink_fire_resist_potion",
    "drink_speed_potion",
    # -- Enchantments (2) --
    "enchant_weapon",
    "enchant_armor",
    # -- Dungeon progression (6) --
    "enter_dungeon",
    "reach_floor_2",
    "reach_floor_3",
    "reach_floor_4",
    "reach_floor_5",
    "return_to_surface",
    # -- Bosses (5) --
    "defeat_knight",
    "defeat_archer_boss",
    "defeat_mage",
    "defeat_dragon",
    "defeat_lich",
    # -- New mobs (3) --
    "defeat_spider",
    "defeat_bat",
    "defeat_skeleton_archer",
    # -- Boss loot (5) --
    "collect_boss_loot_1",
    "collect_boss_loot_2",
    "collect_boss_loot_3",
    "collect_boss_loot_4",
    "collect_boss_loot_5",
    # -- Survival milestones (2) --
    "survive_10_nights",
    "survive_20_nights",
    # -- Stat milestones (2) --
    "full_health",
    "full_mana",
    # -- Kill milestones (3) --
    "kill_10_mobs",
    "kill_25_mobs",
    "kill_50_mobs",
    # -- Craft milestones (2) --
    "craft_10_items",
    "craft_25_items",
    # -- Misc milestones (3) --
    "place_10_blocks",
    "eat_5_plants",
    "drink_10_water",
    # -- Exploration milestones (4) --
    "explore_all_floors",
    "collect_all_boss_loot",
    "learn_all_spells",
    "max_inventory_wood",
    # -- Combat milestones (4) --
    "kill_5_mobs",
    "survive_5_nights",
    "defeat_boss_no_armor",
    "clear_dungeon_floor",
    # -- Resource milestones (3) --
    "collect_10_wood",
    "collect_5_stone",
    "collect_3_iron",
    # -- Total: 80 --
)

# Visible window dimensions
VIEW_WIDTH = 9
VIEW_HEIGHT = 7

# Full version uses slightly larger view
FULL_VIEW_WIDTH = 11
FULL_VIEW_HEIGHT = 9
