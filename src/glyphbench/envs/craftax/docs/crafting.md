# Craftax — Crafting

> Canonical anchors (locked API): `crafting:wood`, `crafting:stone`, `crafting:iron`, `crafting:diamond`, `crafting:placement`, `crafting:arrows`, `crafting:torches`.

Most crafting actions require adjacency (including diagonal) to a placed crafting table (`t`). Iron-tier and above also require an adjacent furnace (`f`). Enchanting requires an enchantment table (`Ⓔ` fire or `Ⓘ` ice).

<!-- :section crafting:placement -->
## Placement actions

| Action | Cost | Prerequisite | Result |
|---|---|---|---|
| PLACE_STONE | 1 stone | none | Places `=` (placed stone) in faced cell |
| PLACE_TABLE | 2 wood | none | Places `t` (crafting table) |
| PLACE_FURNACE | 4 stone | none | Places `f` (furnace) |
| PLACE_PLANT | 1 sapling | none | Places `+` sapling; ripens to `*` after 20 steps |
| PLACE_TORCH | 1 crafted torch | none | Places `!` torch; emits light radius 5 |

PLACE_TORCH consumes from the `torch` inventory key (filled by MAKE_TORCH); not from raw wood or coal.
<!-- :end -->

<!-- :section crafting:wood -->
## Wood-tier crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_WOOD_PICKAXE | 1 wood | adjacent table | Wood pickaxe (mines stone, coal; tier 0) |
| MAKE_WOOD_SWORD | 1 wood | adjacent table | Wood sword (+1 melee bonus) |

Tier 0 pickaxes can mine stone and coal. To advance, mine stone and craft a stone pickaxe.
<!-- :end -->

<!-- :section crafting:stone -->
## Stone-tier crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_STONE_PICKAXE | 1 wood + 1 stone | adjacent table | Stone pickaxe (mines iron; tier 1) |
| MAKE_STONE_SWORD | 1 wood + 1 stone | adjacent table | Stone sword (+2 melee bonus) |

Stone tier requires **only** a crafting table (no furnace needed). Iron and above need both.
<!-- :end -->

<!-- :section crafting:iron -->
## Iron-tier crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_IRON_PICKAXE | 1 wood + 1 iron | adjacent table + furnace | Iron pickaxe (mines diamond, sapphire, ruby; tier 2) |
| MAKE_IRON_SWORD | 1 wood + 1 iron | adjacent table + furnace | Iron sword (+3 melee bonus) |
| MAKE_IRON_ARMOR | 2 iron | adjacent table + furnace | Iron armor piece (tier 1; fills lowest empty slot) |

Iron pickaxe is the gating item for diamond, sapphire, and ruby. Stock coal before iron-tier crafting (furnace placement uses 4 stone; the iron pickaxe recipe consumes wood + iron).
<!-- :end -->

<!-- :section crafting:diamond -->
## Diamond-tier crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_DIAMOND_PICKAXE | 1 wood + 1 diamond | adjacent table + furnace | Diamond pickaxe (tier 3; strongest) |
| MAKE_DIAMOND_SWORD | 1 wood + 1 diamond | adjacent table + furnace | Diamond sword (+4 melee bonus) |
| MAKE_DIAMOND_ARMOR | 1 diamond + 1 iron | adjacent table + furnace | Diamond armor (tier 2; fills or upgrades lowest slot) |

Diamond armor pieces upgrade existing iron-tier slots when all 4 slots are already filled.

**Resource mining prerequisites:**

| Ore | Minimum pickaxe tier |
|---|---|
| Stone | 0 (any pickaxe) |
| Coal | 1 (stone+) |
| Iron | 1 (stone+) |
| Diamond | 2 (iron+) |
| Sapphire (`♦`) | 2 (iron+) |
| Ruby (`▲`) | 2 (iron+) |
<!-- :end -->

<!-- :section crafting:arrows -->
## Arrow crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_ARROW | 1 wood + 1 stone | adjacent table | 2 arrows |

Arrows are consumed one per SHOOT_ARROW. The bow itself is **not crafted** — it drops from the first chest opened on floor 1. See `items:bow`.
<!-- :end -->

<!-- :section crafting:torches -->
## Torch crafting

| Action | Inputs | Prerequisite | Output |
|---|---|---|---|
| MAKE_TORCH | 1 wood + 1 coal | adjacent table | 4 crafted torches |

Crafted torches go into the `torch` inventory slot. PLACE_TORCH consumes 1 to place a `!` tile. Torches are critical for visibility on dark floors (2 Gnomish Mines, 5 Troll Mines, 7 Ice Realm, 8 Graveyard).
<!-- :end -->
