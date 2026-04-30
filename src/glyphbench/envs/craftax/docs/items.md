# Craftax — Items

> Canonical anchors (locked API): `items:resources`, `items:gems`, `items:potions`, `items:bow`, `items:torches`.

<!-- :section items:resources -->
## Resources

Resources are mineable/harvestable inventory items used as crafting inputs:

| Item | Source | Tier required |
|---|---|---|
| wood | DO on tree (`♣`) | none |
| stone | DO on stone (`S`) | wood pickaxe |
| coal | DO on coal ore (`C`) | stone pickaxe |
| iron | DO on iron ore (`I`) | stone pickaxe |
| diamond | DO on diamond ore (`D`) | iron pickaxe |
| sapling | DO on grass (`·`); ~30% chance | none |

Inventory counts appear in the HUD. The `_max_inventory_wood` achievement fires at high wood counts (upstream rule). Wood is the most-used input — table, pickaxes, swords, arrows, torches all consume it.

Saplings drop from grass tiles only on the surface (floor 0). Stock food (cows, plants) before long dungeon runs since saplings cannot be obtained underground.
<!-- :end -->

<!-- :section items:gems -->
## Gems (sapphire and ruby)

| Gem | Glyph | Use | Where to find |
|---|---|---|---|
| Sapphire | `♦` | Ice enchantments at `Ⓘ` ice table | floor 2 (2.5%), floor 5 (rare), floor 7 (2%) |
| Ruby | `▲` | Fire enchantments at `Ⓔ` fire table | floor 2 (2.5%), floor 5 (rare), floor 6 (2.5%) |

Both gems require an iron pickaxe (tier 2) to mine. Each enchant action consumes 1 gem + 9 mana. See `magic:enchants`.

Achievements: `collect_sapphire`, `collect_ruby`.
<!-- :end -->

<!-- :section items:potions -->
## Potions

There are 6 potion colors: red, green, blue, pink, cyan, yellow.

**Per-game hidden shuffle:** at episode start the game assigns each color to one of 6 effects via a random permutation. The mapping is never shown to the agent; identify colors by trial and error.

**Effects** (one per color slot per the hidden permutation):

| Effect ID | Description |
|---|---|
| heal_8 | +8 HP (capped at max HP) |
| poison_3 | -3 HP (treated as damage; armor + sleep multipliers apply) |
| mana_8 | +8 mana (capped at max mana) |
| mana_drain_3 | -3 mana (floor 0) |
| energy_8 | +8 energy (capped at max energy) |
| energy_drain_3 | -3 energy (floor 0) |

Drink actions: DRINK_POTION_RED, DRINK_POTION_GREEN, DRINK_POTION_BLUE, DRINK_POTION_PINK, DRINK_POTION_CYAN, DRINK_POTION_YELLOW. Each consumes 1 potion of the named color.

Potions are found in chests (standard loot rolls). First-drink fires the `drink_potion` achievement. Color → effect mapping changes every episode.
<!-- :end -->

<!-- :section items:bow -->
## Bow

The bow is **not craftable**. It drops from the **first chest opened on floor 1** (achievement `find_bow`).

SHOOT_ARROW requires bow in inventory plus at least 1 arrow (`inventory["arrows"] >= 1`). Arrow damage scales with DEX (see `combat:ranged_player`).

An enchanted bow adds an elemental component (fire or ice) to fired arrows. Enchanting the bow requires the bow in inventory plus 9 mana and 1 gemstone, adjacent to an enchantment table (see `magic:enchants`).
<!-- :end -->

<!-- :section items:torches -->
## Torches

Crafted torches (`inventory["torch"]`) are produced by MAKE_TORCH (1 wood + 1 coal → 4 torches at a table).

PLACE_TORCH consumes 1 from `inventory["torch"]` and places a `!` tile in the faced cell. The placed torch emits light at radius 5 via the lightmap subsystem.

Stock torches before descending to dark floors (2 Gnomish Mines, 5 Troll Mines, 7 Ice Realm, 8 Graveyard). Without torches these floors are nearly invisible.

## Sapling and plants

PLACE_PLANT consumes 1 sapling from inventory and places a `+` tile. After 20 steps the sapling becomes a ripe plant (`*`). EAT_PLANT (or DO while facing `*`) eats it for food. Saplings only grow on the surface (floor 0).

## Chest loot table (general)

Chests (`$`) are opened with DO. Each chest opens exactly once per episode. Standard loot may include wood, torches, iron ore, diamond, a potion (random color), arrows, an iron pickaxe, or an iron sword.

First-chest grants override the standard table on specific floors:
- Floor 1 first chest → bow.
- Floor 3 first chest → book.
- Floor 4 first chest → book (only if floor 3 chest is unopened).

Fountains (`⊙`) are not chests; DO at a fountain refills water.
<!-- :end -->
