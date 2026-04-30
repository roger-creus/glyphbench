"""Coverage registry: action name -> covering anchor names; glyph -> anchors.

Used by tests/glyphbench/envs/craftax/test_tutorial.py to assert that every
env's tutorial_sections covers every action and glyph the env exposes.

When new actions or glyphs are added to the suite, extend the maps below.
The tests will fail loudly if a registered env exposes an action/glyph
that no anchor covers.
"""
from __future__ import annotations

# Action name -> set of anchors that explain this action. An env including
# any of the listed anchors counts as covered for that action.
ACTION_TO_ANCHORS: dict[str, frozenset[str]] = {
    # Self-explanatory; covered by overview (always included by every env).
    "NOOP": frozenset({"overview", "legend:player"}),
    "MOVE_LEFT": frozenset({"overview", "legend:player"}),
    "MOVE_RIGHT": frozenset({"overview", "legend:player"}),
    "MOVE_UP": frozenset({"overview", "legend:player"}),
    "MOVE_DOWN": frozenset({"overview", "legend:player"}),
    # DO is the universal interaction verb (mine, attack, eat, drink, open).
    "DO": frozenset({
        "overview",
        "legend:player",
        "combat:melee",
        "survival:hp_food_drink",
    }),
    # Survival
    "SLEEP": frozenset({"survival:energy_sleep"}),
    "REST": frozenset({"survival:rest"}),
    "EAT_PLANT": frozenset({"survival:hp_food_drink"}),
    "DRINK_WATER": frozenset({"survival:hp_food_drink"}),
    # Placement
    "PLACE_STONE": frozenset({"crafting:placement"}),
    "PLACE_TABLE": frozenset({"crafting:placement"}),
    "PLACE_FURNACE": frozenset({"crafting:placement"}),
    "PLACE_PLANT": frozenset({"crafting:placement"}),
    "PLACE_TORCH": frozenset({"crafting:placement", "crafting:torches"}),
    # Crafting tiers
    "MAKE_WOOD_PICKAXE": frozenset({"crafting:wood"}),
    "MAKE_WOOD_SWORD": frozenset({"crafting:wood"}),
    "MAKE_STONE_PICKAXE": frozenset({"crafting:stone"}),
    "MAKE_STONE_SWORD": frozenset({"crafting:stone"}),
    "MAKE_IRON_PICKAXE": frozenset({"crafting:iron"}),
    "MAKE_IRON_SWORD": frozenset({"crafting:iron"}),
    "MAKE_IRON_ARMOR": frozenset({"crafting:iron"}),
    "MAKE_DIAMOND_PICKAXE": frozenset({"crafting:diamond"}),
    "MAKE_DIAMOND_SWORD": frozenset({"crafting:diamond"}),
    "MAKE_DIAMOND_ARMOR": frozenset({"crafting:diamond"}),
    "MAKE_ARROW": frozenset({"crafting:arrows"}),
    "MAKE_TORCH": frozenset({"crafting:torches"}),
    # Combat (ranged player)
    "SHOOT_ARROW": frozenset({"combat:ranged_player", "items:bow"}),
    # Magic
    "CAST_FIREBALL": frozenset({"magic:spells"}),
    "CAST_ICEBALL": frozenset({"magic:spells"}),
    "READ_BOOK": frozenset({"magic:books"}),
    # Enchanting
    "ENCHANT_WEAPON": frozenset({"magic:enchants"}),
    "ENCHANT_ARMOR": frozenset({"magic:enchants"}),
    "ENCHANT_BOW": frozenset({"magic:enchants"}),
    # Floor navigation
    "DESCEND": frozenset({"floors:navigation"}),
    "ASCEND": frozenset({"floors:navigation"}),
    # Potions (one shared anchor for all 6 colors)
    "DRINK_POTION_RED": frozenset({"items:potions"}),
    "DRINK_POTION_GREEN": frozenset({"items:potions"}),
    "DRINK_POTION_BLUE": frozenset({"items:potions"}),
    "DRINK_POTION_PINK": frozenset({"items:potions"}),
    "DRINK_POTION_CYAN": frozenset({"items:potions"}),
    "DRINK_POTION_YELLOW": frozenset({"items:potions"}),
    # Attributes
    "LEVEL_UP_DEXTERITY": frozenset({"progression:attributes"}),
    "LEVEL_UP_STRENGTH": frozenset({"progression:attributes"}),
    "LEVEL_UP_INTELLIGENCE": frozenset({"progression:attributes"}),
}


# Glyph (Unicode codepoint) -> anchors that mention it. An env's
# tutorial_sections must include at least one anchor for every glyph
# the env can render on its grid.
GLYPH_TO_ANCHORS: dict[str, frozenset[str]] = {
    # Player (rendered as directional arrow; @ is the fallback)
    "@": frozenset({"legend:player"}),
    "→": frozenset({"legend:player"}),
    "←": frozenset({"legend:player"}),
    "↑": frozenset({"legend:player"}),
    "↓": frozenset({"legend:player"}),

    # Surface terrain
    "·": frozenset({"legend:terrain"}),  # · grass
    "♣": frozenset({"legend:terrain"}),  # ♣ tree
    "S": frozenset({"legend:terrain"}),
    "C": frozenset({"legend:terrain"}),
    "I": frozenset({"legend:terrain"}),
    "D": frozenset({"legend:terrain"}),
    "≈": frozenset({"legend:terrain"}),  # ≈ water
    "♨": frozenset({"legend:terrain"}),  # ♨ lava
    "░": frozenset({"legend:terrain"}),  # ░ sand
    "=": frozenset({"legend:terrain"}),       # placed stone
    "+": frozenset({"legend:terrain"}),       # planted sapling
    "*": frozenset({"legend:terrain"}),       # ripe plant

    # Dungeon terrain
    "█": frozenset({"legend:terrain"}),  # █ dungeon wall
    "▪": frozenset({"legend:terrain"}),  # ▪ dungeon floor

    # Biome decorations
    "♠": frozenset({"legend:terrain"}),  # ♠ fire tree (floor 6)
    "❄": frozenset({"legend:terrain"}),  # ❄ ice shrub (floor 7)
    "⚰": frozenset({"legend:terrain"}),  # ⚰ grave (floor 8)

    # Items / interactives
    "t": frozenset({"legend:items", "crafting:placement"}),
    "f": frozenset({"legend:items", "crafting:placement"}),
    "!": frozenset({"legend:items", "crafting:placement", "crafting:torches"}),
    "B": frozenset({"legend:items"}),  # boss door
    "$": frozenset({"legend:items", "items:bow", "magic:books"}),
    "⊙": frozenset({"legend:items", "survival:hp_food_drink"}),  # ⊙ fountain
    "Ⓔ": frozenset({"legend:items", "magic:enchants", "floors:4"}),  # Ⓔ
    "Ⓘ": frozenset({"legend:items", "magic:enchants", "floors:3"}),  # Ⓘ
    ";": frozenset({"legend:items"}),  # sapling-in-inventory tile (placed)
    "⇣": frozenset({"floors:navigation"}),  # ⇣ stairs down
    "⇡": frozenset({"floors:navigation"}),  # ⇡ stairs up

    # Gem ores
    "♦": frozenset({"items:gems"}),  # ♦ sapphire
    "▲": frozenset({"items:gems"}),  # ▲ ruby

    # Mobs - overworld
    "z": frozenset({"legend:mobs:overworld"}),
    "c": frozenset({"legend:mobs:overworld"}),
    "a": frozenset({"legend:mobs:overworld", "legend:mobs:dungeon"}),  # skeleton + knight_archer

    # Mobs - dungeon
    "q": frozenset({"legend:mobs:dungeon"}),  # kobold
    "b": frozenset({"legend:mobs:dungeon"}),  # bat
    "s": frozenset({"legend:mobs:dungeon"}),  # snail
    "T": frozenset({"legend:mobs:dungeon"}),  # troll
    "d": frozenset({"legend:mobs:dungeon"}),  # deep thing
    "p": frozenset({"legend:mobs:dungeon"}),  # pigman
    "F": frozenset({"legend:mobs:dungeon"}),  # fire elemental
    "r": frozenset({"legend:mobs:dungeon"}),  # frost troll
    "i": frozenset({"legend:mobs:dungeon"}),  # ice elemental
    "W": frozenset({"legend:mobs:dungeon"}),  # legacy boss

    # Boss
    "N": frozenset({"legend:mobs:boss"}),
    "n": frozenset({"legend:mobs:boss"}),

    # Projectiles
    "↗": frozenset({"legend:projectiles", "combat:projectiles"}),  # ↗ arrow1
    "↘": frozenset({"legend:projectiles", "combat:projectiles"}),  # ↘ arrow2
    "†": frozenset({"legend:projectiles", "combat:projectiles"}),  # † dagger
    "●": frozenset({"legend:projectiles", "combat:projectiles"}),  # ● fireball1
    "◉": frozenset({"legend:projectiles", "combat:projectiles"}),  # ◉ fireball2
    "○": frozenset({"legend:projectiles", "combat:projectiles"}),  # ○ iceball1
    "◎": frozenset({"legend:projectiles", "combat:projectiles"}),  # ◎ iceball2
    "◐": frozenset({"legend:projectiles", "combat:projectiles"}),  # ◐ slimeball
}
