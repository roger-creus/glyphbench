# Craftax — Overview

> Canonical anchors (locked API): `overview`.

<!-- :section overview -->
Craftax is a 9-floor survival crafting game. The agent gathers resources, crafts tools and weapons, explores progressively harder dungeon floors, learns magic, and ultimately defeats the Necromancer on floor 8. Each floor is harder than the last; floor 8 is the final boss floor with no exit.

## Floor stack

| Floor | Name | Biome |
|---|---|---|
| 0 | Overworld | Smoothgen 64x64 |
| 1 | Dungeon | Dungeon-rooms 32x32 |
| 2 | Gnomish Mines | Smoothgen 32x32 |
| 3 | Sewers | Dungeon-rooms 32x32 |
| 4 | Vaults | Dungeon-rooms 32x32 |
| 5 | Troll Mines | Smoothgen 32x32 |
| 6 | Fire Realm | Smoothgen 32x32 |
| 7 | Ice Realm | Smoothgen 32x32 |
| 8 | Graveyard | Smoothgen 32x32 |

See `floors.md` for per-floor mob rosters, special tiles, and transition rules.

## Action structure

The full action spec has 45 named actions. Key categories:

- **Movement**: MOVE_LEFT / MOVE_RIGHT / MOVE_UP / MOVE_DOWN
- **Interaction**: DO (face target and interact: mine, attack, open chest, drink, eat)
- **Placement**: PLACE_STONE / PLACE_TABLE / PLACE_FURNACE / PLACE_PLANT / PLACE_TORCH
- **Crafting**: MAKE_WOOD/STONE/IRON/DIAMOND_PICKAXE; MAKE_WOOD/STONE/IRON/DIAMOND_SWORD; MAKE_IRON_ARMOR; MAKE_DIAMOND_ARMOR; MAKE_ARROW; MAKE_TORCH
- **Combat**: SHOOT_ARROW; CAST_FIREBALL; CAST_ICEBALL; ENCHANT_WEAPON; ENCHANT_ARMOR; ENCHANT_BOW
- **Survival**: SLEEP; REST; EAT_PLANT; DRINK_WATER; DRINK_POTION_RED/GREEN/BLUE/PINK/CYAN/YELLOW
- **Exploration**: DESCEND; ASCEND; READ_BOOK
- **Progression**: LEVEL_UP_DEXTERITY; LEVEL_UP_STRENGTH; LEVEL_UP_INTELLIGENCE; NOOP

## Reward

- **+1** per first-time achievement unlock (93 achievements total).
- **+10** for defeating the Necromancer (win condition).
- No other reward signals exist. Achievements are the primary optimization target.

## Win and death

- **Death**: HP reaches 0. Episode terminates with no bonus.
- **Win**: `boss_progress >= 8` (8 hits on the vulnerable Necromancer). Episode terminates with +10 reward and the `defeat_necromancer` achievement.

## Observation conventions

Each turn the agent receives:

- **Grid**: ASCII art of the 11x9 viewport centered on the agent. Each cell is one Unicode codepoint.
- **Legend**: per-turn glyph key listing every glyph visible in the current grid (see `legend.md` for the full palette).
- **HUD**: HP / Food / Drink / Energy / Mana / XP / DEX / STR / INT / Floor / Step, plus the full inventory and armor slots.
- **Message**: last game message (achievement unlocks, damage events, crafting results, etc.).

The agent selects one action name per turn. Invalid or blocked actions are silently no-oped.
<!-- :end -->
