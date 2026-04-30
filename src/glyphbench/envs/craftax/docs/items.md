# Craftax — Items

## Bow and arrows

**Bow**: The bow is not craftable. It drops from the **first chest opened on floor 1**. The `find_bow` achievement fires on acquisition.

**Arrows**: Craftable. Recipe: `MAKE_ARROW` = 1 wood + 1 stone at a crafting table → yields 2 arrows.

**SHOOT_ARROW**: Fires 1 arrow in the facing direction. Requires:
- Bow in inventory (`inventory["bow"] >= 1`).
- At least 1 arrow (`inventory["arrows"] >= 1`).

Arrow damage is physical (no element) and scales with DEX:

| DEX | Arrow damage multiplier |
|---|---|
| 1 | 1.0× |
| 2 | 1.2× |
| 3 | 1.4× |
| 4 | 1.6× |
| 5 | 1.8× |

The arrow projectile uses glyph `↗` or `↘` depending on variant. Collision and travel mechanics follow the standard projectile rules (see `combat.md`).

An enchanted bow adds the bow's elemental component (fire or ice) to each arrow. Enchanting the bow requires the bow in inventory plus 9 mana and a gemstone adjacent to an enchantment table (see `crafting.md`).

## Torches

**MAKE_TORCH**: Crafts 4 torches. Costs 1 wood + 1 coal at a crafting table. Torches go into the `torch` inventory key.

**PLACE_TORCH**: Consumes 1 crafted torch from `inventory["torch"]` and places a `!` tile in the faced cell. The placed torch emits light at radius 5, computed via the lightmap subsystem. Torches are critical for visibility in dark dungeon floors (2, 5, 6, 7, 8).

Stock torches before descending to floor 2 (Gnomish Mines); the floor is dark without them.

## Potions

There are 6 potion colors: **red, green, blue, pink, cyan, yellow**.

**Per-game hidden shuffle**: at episode start the game assigns each color to one of 6 effects via a random permutation. The mapping is never shown to the agent. The agent must identify colors by trial and error.

**Effects** (one per color slot, per the hidden permutation):

| Effect ID | Description |
|---|---|
| heal_8 | +8 HP (capped at max HP) |
| poison_3 | -3 HP (treated as damage, applies armor/sleep multipliers) |
| mana_8 | +8 mana (capped at max mana) |
| mana_drain_3 | -3 mana (floor 0) |
| energy_8 | +8 energy (capped at max energy) |
| energy_drain_3 | -3 energy (floor 0) |

**Drink actions**: DRINK_POTION_RED / DRINK_POTION_GREEN / DRINK_POTION_BLUE / DRINK_POTION_PINK / DRINK_POTION_CYAN / DRINK_POTION_YELLOW. Each consumes 1 potion of the named color.

Potions are found in chests (standard loot rolls). A first-drink of any potion fires the `drink_potion` achievement. The color→effect mapping changes every episode; do not assume a fixed mapping across runs.

## Gems

**Sapphire** (`♦`): mined with an iron pickaxe (tier 2+). Appears on floors 2 (2.5%), 5 (rare), 7 (2%). Used for **ice enchantments** (ENCHANT_SWORD / ENCHANT_ARMOR / ENCHANT_BOW with sapphire + 9 mana at an ice table).

**Ruby** (`▲`): mined with an iron pickaxe (tier 2+). Appears on floors 2 (2.5%), 5 (rare), 6 (2.5%). Used for **fire enchantments** (ENCHANT_SWORD / ENCHANT_ARMOR / ENCHANT_BOW with ruby + 9 mana at a fire table).

Gems are consumed on use (1 gem per enchant action). See `crafting.md` for enchanting prerequisites.

## Books

Books are quest items that teach spells. They appear only from first-chest grants:

- Floor 3 first chest: grants 1 book.
- Floor 4 first chest: grants 1 book (only if floor 3 first chest was not yet opened).

Reading a book (READ_BOOK action) consumes it and randomly teaches one unlearned spell (fireball or iceball). See `magic.md` for spell details.

## Chest loot table

Chests (`$` glyph) are opened with DO while facing the chest. Each chest opens exactly once per episode. The loot roll follows an upstream-faithful distribution and may yield:

- Wood (common)
- Torches
- Iron ore
- Diamond
- Potion (one of the 6 colors)
- Arrows
- Iron pickaxe
- Iron sword

First-chest grants override the normal loot table for the specific floor-priority items (bow on floor 1, book on floors 3/4). All other chests on those floors roll from the standard table.

Fountains (`⊙`) are not chests; they refill water when interacted with via DO.

## Sapling and plants

**Sapling** (`;` — placed with PLACE_PLANT from `inventory["sapling"]`): saplings drop from grass tiles with ~30% chance when DO is used on a grass tile on the surface. Plant a sapling and return after 20 steps to find a ripe plant (`*`).

**EAT_PLANT**: DO while facing a ripe plant (`*`) eats it, restoring food. The eaten tile reverts to grass.

Saplings do not grow in dungeons. Stock food (or cows/snails) before long dungeon runs.
