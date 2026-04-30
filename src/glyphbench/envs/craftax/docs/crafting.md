# Craftax — Crafting

## Adjacency prerequisites

Most crafting actions require the player to be **adjacent** (including diagonally) to a placed crafting table (`t`). Some require a furnace (`f`) in addition. Enchanting requires an enchantment table.

| Prerequisite label | What must be adjacent |
|---|---|
| table | Crafting table (`t`) |
| table+furnace | Both table (`t`) AND furnace (`f`) |
| FIRE table | Fire enchantment table (`Ⓔ`, floor 4) |
| ICE table | Ice enchantment table (`Ⓘ`, floor 3) |
| FIRE table OR ICE table | Either enchantment table |

Diagonally adjacent counts for table+furnace requirements.

## Placement actions

| Action | Cost | Prerequisite | Result |
|---|---|---|---|
| PLACE_STONE | 1 stone | none | Places `=` (placed stone) in faced cell |
| PLACE_TABLE | 2 wood | none | Places `t` (crafting table) in faced cell |
| PLACE_FURNACE | 4 stone | none | Places `f` (furnace) in faced cell |
| PLACE_PLANT | 1 sapling | none | Places `+` (sapling) in faced cell; ripens to `*` after 20 steps |
| PLACE_TORCH | 1 crafted torch | none | Places `!` (torch) in faced cell; emits light radius 5 |

Crafted torches come from MAKE_TORCH (see below). Do not confuse them with coal torches — PLACE_TORCH consumes the `torch` inventory key, not wood or coal directly.

## Pickaxe crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_WOOD_PICKAXE | 1 wood | table | Wood pickaxe (mines stone, coal; tier 0) |
| MAKE_STONE_PICKAXE | 1 wood + 1 stone | table | Stone pickaxe (mines iron; tier 1) |
| MAKE_IRON_PICKAXE | 1 wood + 1 iron | table+furnace | Iron pickaxe (mines diamond, sapphire, ruby; tier 2) |
| MAKE_DIAMOND_PICKAXE | 1 wood + 1 diamond | table+furnace | Diamond pickaxe (tier 3; strongest) |

Pickaxe tier determines which ores can be mined. Stone requires tier 0+; coal and iron require tier 1+; diamond, sapphire, ruby require tier 2+ (iron pickaxe or better).

## Sword crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_WOOD_SWORD | 1 wood | table | Wood sword (+1 damage bonus) |
| MAKE_STONE_SWORD | 1 wood + 1 stone | table | Stone sword (+2 damage bonus) |
| MAKE_IRON_SWORD | 1 wood + 1 iron | table+furnace | Iron sword (+3 damage bonus) |
| MAKE_DIAMOND_SWORD | 1 wood + 1 diamond | table+furnace | Diamond sword (+4 damage bonus) |

Only the highest-tier sword held is used in combat. See `combat.md` for the full damage formula.

## Armor crafting

Armor occupies 4 independent slots: helmet, chest, legs, boots. Each crafting action fills the **lowest empty slot** (or upgrades the lowest slot if all are filled with a lower tier).

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_IRON_ARMOR | 2 iron | table+furnace | Iron armor piece (tier 1; fills lowest empty slot) |
| MAKE_DIAMOND_ARMOR | 1 diamond + 1 iron | table+furnace | Diamond armor piece (tier 2; fills or upgrades lowest slot) |

Each filled slot provides 0.1 physical defense reduction. Enchanting adds elemental defense per slot. See `combat.md` for the defense formula.

## Ranged crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_ARROW | 1 wood + 1 stone | table | 2 arrows |

Arrows are consumed one per SHOOT_ARROW action. The bow is found in chests, not crafted. See `items.md`.

## Torch crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_TORCH | 1 wood + 1 coal | table | 4 crafted torches |

Crafted torches go into the `torch` inventory slot and are consumed by PLACE_TORCH.

## Enchanting

Enchanting requires adjacency to an enchantment table, a gemstone, and mana.

| Action | Inputs | Prerequisite | Effect |
|---|---|---|---|
| ENCHANT_WEAPON | 1 ruby (fire) OR 1 sapphire (ice) + 9 mana | FIRE table (floor 4) OR ICE table (floor 3) | Sets sword enchantment to fire or ice element |
| ENCHANT_ARMOR | 1 ruby (fire) OR 1 sapphire (ice) + 9 mana | FIRE table OR ICE table | Sets fire/ice enchant on lowest unenchanted armor slot |
| ENCHANT_BOW | 1 ruby (fire) OR 1 sapphire (ice) + 9 mana | FIRE table OR ICE table (requires bow in inventory) | Sets bow enchantment to fire or ice element |

- Ruby = fire enchantment. Sapphire = ice enchantment.
- An enchanted sword adds `0.5 × physical_damage` in the enchanted element.
- An enchanted armor slot adds 0.2 resistance to the matching element per slot.
- An enchanted bow adds the elemental component to fired arrows.
- See `combat.md` for full damage/defense formulas.

## READ_BOOK

| Action | Inputs | Prerequisite | Effect |
|---|---|---|---|
| READ_BOOK | 1 book | book in inventory | Teaches one unlearned spell (fireball or iceball, randomly) |

Books are found in chests on floors 3 and 4. See `items.md` and `magic.md`.

## Resource mining prerequisites

| Ore | Minimum pickaxe tier | Notes |
|---|---|---|
| Stone | 0 (any) | Mine with bare wood pickaxe |
| Coal | 1 (stone+) | Stone pickaxe required |
| Iron | 1 (stone+) | Stone pickaxe required |
| Diamond | 2 (iron+) | Iron pickaxe required |
| Sapphire (♦) | 2 (iron+) | Iron pickaxe required; floors 2, 5, 7 |
| Ruby (▲) | 2 (iron+) | Iron pickaxe required; floors 2, 5, 6 |
