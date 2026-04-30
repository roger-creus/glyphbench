# Craftax — Legend

The ASCII renderer uses a single Unicode codepoint per grid cell. Colors are present in upstream Craftax renders but this port uses single-codepoint glyphs only — color encoding is not used; the glyph alone identifies the tile.

## Terrain tiles

| Glyph | Name | Category |
|---|---|---|
| `·` | Grass | terrain |
| `♣` | Tree (choppable) | terrain |
| `S` | Stone (mineable) | terrain |
| `C` | Coal ore | terrain |
| `I` | Iron ore | terrain |
| `D` | Diamond ore | terrain |
| `♦` | Sapphire ore | terrain |
| `▲` | Ruby ore | terrain |
| `≈` | Water | terrain |
| `♨` | Lava | terrain |
| `░` | Sand | terrain |
| `█` | Dungeon wall | terrain |
| `▪` | Dungeon floor | terrain |
| `=` | Placed stone | terrain |

## Special / interactive tiles

| Glyph | Name | Category |
|---|---|---|
| `⇣` | Stairs down | special |
| `⇡` | Stairs up | special |
| `t` | Crafting table | special |
| `f` | Furnace | special |
| `!` | Placed torch | special |
| `B` | Boss door | special |
| `$` | Chest (lootable) | special |
| `⊙` | Fountain (drink) | special |
| `Ⓔ` | Fire enchant table (floor 4) | special |
| `Ⓘ` | Ice enchant table (floor 3) | special |
| `♠` | Fire tree (floor 6 decoration) | special |
| `❄` | Ice shrub (floor 7 decoration) | special |
| `⚰` | Grave marker (floor 8 decoration) | special |
| `+` | Sapling (planted) | terrain |
| `*` | Ripe plant (edible) | terrain |
| `;` | Sapling (inventory — placed with PLACE_PLANT) | items |

## Mob tiles

| Glyph | Name | Category |
|---|---|---|
| `@` | Player agent | special |
| `z` | Zombie (melee, floor 0) | mobs |
| `c` | Cow (passive, floor 0) | mobs |
| `a` | Skeleton (ranged, floor 0) | mobs |
| `q` | Kobold (ranged, floor 3) | mobs |
| `b` | Bat (passive, floors 2/5/6/7) | mobs |
| `s` | Snail (passive, floors 1/3/4) | mobs |
| `T` | Troll (melee, floor 5) | mobs |
| `d` | Deep thing (ranged, floor 5) | mobs |
| `p` | Pigman (melee, floor 6) | mobs |
| `F` | Fire elemental (ranged, floor 6) | mobs |
| `r` | Frost troll (melee, floor 7) | mobs |
| `i` | Ice elemental (ranged, floor 7) | mobs |
| `W` | Boss (legacy floor bosses) | mobs |
| `N` | Necromancer (invulnerable) | mobs |
| `n` | Necromancer (vulnerable — can be hit) | mobs |

Note: gnome_warrior, gnome_archer, orc_soldier, orc_mage, lizard, knight, knight_archer use the upstream tile mapping from mobs.py; the glyphs above cover all tiles defined in base.py.

## Projectile tiles

| Glyph | Name | Category |
|---|---|---|
| `↗` | Arrow (variant 1) | projectiles |
| `↘` | Arrow (variant 2) | projectiles |
| `†` | Dagger (kobold) | projectiles |
| `●` | Fireball (variant 1) | projectiles |
| `◉` | Fireball (variant 2, fire elemental) | projectiles |
| `○` | Iceball (variant 1) | projectiles |
| `◎` | Iceball (variant 2, ice elemental) | projectiles |
| `◐` | Slimeball (deep thing) | projectiles |

## HUD fields

The HUD shows the following fields every turn:

| Field | Description |
|---|---|
| HP | Current / max hit points |
| Food | Food level (0 = HP drains) |
| Drink | Water level (0 = HP drains) |
| Energy | Energy (0 = 50% move failure) |
| Mana | Mana for spells and enchanting |
| XP | Unspent experience points |
| DEX | Dexterity attribute (1-5) |
| STR | Strength attribute (1-5) |
| INT | Intelligence attribute (1-5) |
| Floor | Current floor number (0-8) |
| Step | Turn counter |
| Inventory | Counts for every item held |
| Armor | Per-slot tier + enchant element |
| Achievements | Upstream bitmap count (X / 67) |

For per-floor tile details see `floors.md`. For attribute scaling see `progression.md`.
