# Craftax — Combat

> Canonical anchors (locked API): `combat:melee`, `combat:ranged_player`, `combat:ranged_mob`, `combat:armor`, `combat:projectiles`, `combat:elemental`, `combat:sleep_penalty`.

All damage is a 3-vector `(physical, fire, ice)`. Each element is reduced separately by the corresponding defense fraction before summing to an integer.

<!-- :section combat:melee -->
## Melee (player → mob)

DO while facing a mob attacks it. The base attack always deals physical damage:

```
base_phys = 1 + weapon_bonus
scaled_phys = base_phys × (1 + 0.25 × (STR - 1))
```

**Weapon bonuses (physical tier):**

| Weapon | Bonus |
|---|---|
| Bare fist | 0 |
| Wood sword | +1 |
| Stone sword | +2 |
| Iron sword | +3 |
| Diamond sword | +4 |

**Sword enchantment**: an enchanted sword (fire or ice) additionally deals `0.5 × scaled_phys` in the enchanted element. Example: diamond sword (+4) at STR 3 deals `(1+4) × 1.5 = 7.5` physical; with fire enchant also deals `0.5 × 7.5 = 3.75` fire damage.

## Melee mob AI

Melee mobs (`zombie`, `gnome_warrior`, `orc_soldier`, `lizard`, `knight`, `troll`, `pigman`, `frost_troll`):
1. **Attack cooldown** ticks down by 1 each step. When cooldown ≤ 0 and the mob is adjacent (Manhattan distance = 1), the mob attacks and resets cooldown to **5 ticks**.
2. **Movement** (occurs even while cooldown > 0):
   - Chase (75% probability if Manhattan distance < 10, OR always during boss fight): move one step toward the player along the longer axis.
   - Random walk (25% otherwise): move ±1 in a random cardinal direction or stay.
3. Mobs are blocked by walls, ores, stone, placed stone, and other solid tiles.

Step-loop order: player action → player projectiles advance → mob AI runs → mob projectiles advance → despawn sweep → vitals decay → milestone checks. The player always acts before mobs respond.
<!-- :end -->

<!-- :section combat:ranged_player -->
## Ranged combat (player)

**SHOOT_ARROW** fires 1 arrow in the facing direction. Requirements:
- Bow in inventory (`inventory["bow"] >= 1`).
- At least 1 arrow (`inventory["arrows"] >= 1`).

Arrow damage is physical and scales with DEX: `× (1 + 0.2 × (DEX - 1))`.

| DEX | Arrow damage multiplier |
|---|---|
| 1 | 1.0× |
| 2 | 1.2× |
| 3 | 1.4× |
| 4 | 1.6× |
| 5 | 1.8× |

Arrow projectile glyph: `↗` or `↘`. An enchanted bow adds a fire or ice elemental component to fired arrows. See `magic:enchants` for enchanting.

The bow drops from the first chest opened on floor 1; it cannot be crafted. Arrows are crafted via MAKE_ARROW (1 wood + 1 stone → 2 arrows) at a table.
<!-- :end -->

<!-- :section combat:ranged_mob -->
## Ranged combat (mobs)

Ranged mobs (`skeleton`, `gnome_archer`, `orc_mage`, `kobold`, `knight_archer`, `deep_thing`, `fire_elemental`, `ice_elemental`):
1. **Attack cooldown** ticks down by 1 each step.
2. **Shoot window:** Manhattan distance ∈ [4, 5] AND cooldown ≤ 0 AND projectile slot available AND 85% probability → shoot toward player; reset cooldown to **4 ticks**; no movement this tick.
3. **Cornered fallback:** if distance ≤ 3 AND all 4 cardinal retreat directions are blocked AND cooldown ≤ 0 → force shoot.
4. **Movement:**
   - 15% override: random walk regardless of distance.
   - Distance ≥ 6: advance toward player.
   - Distance ≤ 3: retreat (kiting).
   - 3 < distance < 6 (not in shoot window): random walk.

| Mob | Projectile | Element |
|---|---|---|
| skeleton, gnome_archer, knight_archer | Arrow / Arrow2 | Physical |
| orc_mage, fire_elemental | Fireball / Fireball2 | Fire |
| kobold | Dagger | Physical |
| deep_thing | Slimeball | Mixed (phys/fire/ice equal thirds) |
| ice_elemental | Iceball2 | Ice |

Mobs despawn when their Manhattan distance from the player exceeds 14. Exception: floor 8 boss fight — melee/ranged mobs are exempt from despawn so summoned waves persist until killed.
<!-- :end -->

<!-- :section combat:armor -->
## Armor (player defense)

The player has 4 armor slots: helmet, chest, legs, boots. Each crafting action (MAKE_IRON_ARMOR, MAKE_DIAMOND_ARMOR) fills the **lowest empty slot**.

- Each filled slot grants **0.1 physical defense**.
- Each fire-enchanted slot grants **0.2 fire defense**.
- Each ice-enchanted slot grants **0.2 ice defense**.
- Maximum defense per element: 1.0 (100% reduction).

Examples:
- 4 unenchanted iron/diamond slots → 0.4 phys defense (60% phys passes through).
- 4 fire-enchanted slots → 0.8 fire defense (20% fire passes through).

Mob defense vectors (fraction of incoming damage reduced per element):

| Mob | Phys | Fire | Ice |
|---|---|---|---|
| zombie, skeleton, kobold | 0.0 | 0.0 | 0.0 |
| gnome_warrior, gnome_archer | 0.0 | 0.0 | 0.0 |
| orc_soldier, orc_mage | 0.0 | 0.0 | 0.0 |
| lizard | 0.0 | 0.0 | 0.0 |
| knight, knight_archer (floor 4) | 0.5 | 0.0 | 0.0 |
| troll, deep_thing (floor 5) | 0.2 | 0.0 | 0.0 |
| pigman, fire_elemental (floor 6) | 0.9 | 1.0 | 0.0 |
| frost_troll, ice_elemental (floor 7) | 0.9 | 0.0 | 1.0 |
<!-- :end -->

<!-- :section combat:projectiles -->
## Projectiles

All projectiles (player and mob) travel **1 tile per turn** in the direction of spawn. Collision is checked at both the **pre-advance position** and the **post-advance position** each tick.

A projectile is consumed on:
- Hitting a mob (player projectiles) or the player (mob projectiles).
- Hitting a solid blocking tile (walls, stone, ores, trees, placed stone, boss door, crafting structures, plants).
- Leaving the floor boundaries.

Spawn rules:
- Player projectiles spawn at the agent's tile (the agent fires from where they stand).
- Mob projectiles spawn one tile in front of the mob.

The per-tick driver advances projectiles immediately after the action phase, so a freshly cast spell may hit a target in the same turn it is spawned if the target is adjacent.

Mob projectiles can destroy crafting tables and furnaces on impact.
<!-- :end -->

<!-- :section combat:elemental -->
## Elemental damage typing

Damage types: physical, fire, ice. Each type is reduced by its corresponding mob defense fraction independently.

Fire-realm mobs (floor 6 pigman, fire_elemental) have **fire defense = 1.0** (immunity). Use ice spells/enchanted weapons.

Ice-realm mobs (floor 7 frost_troll, ice_elemental) have **ice defense = 1.0** (immunity). Use fire spells/enchanted weapons.

Floor 4 vault knights (knight, knight_archer) have **0.5 physical defense**. Pure physical builds deal half damage. Use enchanted weapons, spells, or enchanted bows.

**Boss-floor multiplier:** all incoming damage on floor 8 is multiplied by **1.5×** (after armor reduction, before sleep multiplier).

Order of damage application: raw mob damage → armor reduction → boss-floor multiplier → sleep multiplier.
<!-- :end -->

<!-- :section combat:sleep_penalty -->
## Sleep damage penalty

While `_is_sleeping = True`, all incoming damage is multiplied by **3.5×** (after armor reduction, before boss-floor multiplier).

Combined with the floor-8 multiplier:

| Condition | Multiplier |
|---|---|
| Floor 8 only | × 1.5 |
| Sleeping only | × 3.5 |
| Floor 8 + sleeping | × 5.25 |

REST does **not** apply this multiplier. Only `_is_sleeping` activates the 3.5× penalty.
<!-- :end -->
