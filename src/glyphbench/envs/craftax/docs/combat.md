# Craftax — Combat

## Damage model

All damage is a **3-vector** `(physical, fire, ice)`. Each element is reduced separately by the corresponding defense fraction before summing to an integer.

### Player attacking mobs

The base attack always deals physical damage. The formula:

```
base_phys = 1 + weapon_bonus
scaled_phys = base_phys × (1 + 0.25 × (STR - 1))
```

**Weapon bonuses** (physical tier only):

| Weapon | Bonus |
|---|---|
| Bare fist | 0 |
| Wood sword | +1 |
| Stone sword | +2 |
| Iron sword | +3 |
| Diamond sword | +4 |

**Sword enchantment**: an enchanted sword (fire or ice element) additionally deals `0.5 × scaled_phys` in the enchanted element. Example: a diamond sword (bonus +4) at STR 3 deals `(1+4) × 1.5 = 7.5` physical. With a fire enchant it also deals `0.5 × 7.5 = 3.75` fire damage.

**Mob defense vectors** (fraction of incoming damage reduced per element):

| Mob | Phys defense | Fire defense | Ice defense |
|---|---|---|---|
| zombie, skeleton, kobold | 0.0 | 0.0 | 0.0 |
| gnome_warrior, gnome_archer | 0.0 | 0.0 | 0.0 |
| orc_soldier, orc_mage | 0.0 | 0.0 | 0.0 |
| lizard | 0.0 | 0.0 | 0.0 |
| knight, knight_archer (floor 4) | 0.5 | 0.0 | 0.0 |
| troll, deep_thing (floor 5) | 0.2 | 0.0 | 0.0 |
| pigman, fire_elemental (floor 6) | 0.9 | 1.0 | 0.0 |
| frost_troll, ice_elemental (floor 7) | 0.9 | 0.0 | 1.0 |

Fire-realm mobs are immune to fire damage (`defense = 1.0`). Ice-realm mobs are immune to ice damage. Floor 4 vault knights halve physical damage. Use enchanted weapons or spells against these floors.

### Mobs attacking the player

Mobs deal scalar damage that the engine treats as physical. The player's armor then reduces it:

**Player armor defense** (multiplicative per slot):

- Each filled armor slot (tier > 0) grants **0.1 physical defense**.
- Each fire-enchanted armor slot grants **0.2 fire defense**.
- Each ice-enchanted armor slot grants **0.2 ice defense**.
- Maximum defense per element: 1.0 (100% reduction).

With all 4 slots filled and iron/diamond tier: `4 × 0.1 = 0.4` physical defense (60% of physical damage passes through). With all 4 slots fire-enchanted: `4 × 0.2 = 0.8` fire defense (20% of fire passes through).

**Damage multipliers** applied after armor:

| Condition | Multiplier |
|---|---|
| Floor 8 (Graveyard/boss floor) | × 1.5 |
| Sleeping (`_is_sleeping = True`) | × 3.5 |
| Both simultaneously | × 5.25 |

The order is: raw mob damage → armor reduction → boss-floor multiplier → sleep multiplier.

## Melee mob AI

Melee mobs (`zombie`, `gnome_warrior`, `orc_soldier`, `lizard`, `knight`, `troll`, `pigman`, `frost_troll`) use `step_melee_mob`:

1. **Attack cooldown**: ticks down by 1 each step. When cooldown reaches 0 and the mob is adjacent (Manhattan distance = 1), the mob attacks and resets cooldown to **5 ticks**.
2. **Movement** (occurs even while cooldown > 0):
   - **Chase** (75% probability if Manhattan distance < 10, OR always during boss fight): move one step toward the player along the longer axis.
   - **Random walk** (25% otherwise): move ±1 in a random cardinal direction or stay.
3. Mobs are blocked by walls, ores, stone, placed stone, and other solid tiles.

## Ranged mob AI

Ranged mobs (`skeleton`, `gnome_archer`, `orc_mage`, `kobold`, `knight_archer`, `deep_thing`, `fire_elemental`, `ice_elemental`) use `step_ranged_mob`:

1. **Attack cooldown**: ticks down by 1 each step.
2. **Shoot window**: Manhattan distance ∈ [4, 5] AND cooldown ≤ 0 AND projectile slot available AND 85% probability → shoot projectile toward player; reset cooldown to **4 ticks**; no movement this tick.
3. **Cornered fallback** (phase γ): if distance ≤ 3 AND all 4 cardinal retreat directions are blocked AND cooldown ≤ 0 → force shoot even if outside the normal window.
4. **Movement**:
   - 15% override: random walk regardless of distance.
   - Distance ≥ 6: advance toward player.
   - Distance ≤ 3: retreat away from player (kiting).
   - 3 < distance < 6 (not in shoot window): random walk.

Ranged mobs fire their mob-specific projectile type:

| Mob | Projectile | Element |
|---|---|---|
| skeleton, gnome_archer, knight_archer | Arrow / Arrow2 | Physical |
| orc_mage, fire_elemental | Fireball / Fireball2 | Fire |
| kobold | Dagger | Physical |
| deep_thing | Slimeball | Mixed (phys/fire/ice equal thirds) |
| ice_elemental | Iceball2 | Ice |

## Projectiles

All projectiles (player and mob) travel **1 tile per tick** in the direction of spawn. Collision is checked at both the **pre-advance position** and the **post-advance position** each tick (phase γ, resolves T_FOLLOWUP_C). A projectile is consumed on:

- Hitting a mob (player projectiles) or the player (mob projectiles).
- Hitting a solid blocking tile (walls, stone, ores, trees, placed stone, boss door, crafting structures, plants).
- Leaving the floor boundaries.

Projectiles spawn at the **agent's tile** (player) or **one tile in front of the mob** (mob projectiles). The per-tick driver advances the projectile immediately after the action phase, so a cast spell may hit a target in the same turn it is spawned if the target is adjacent.

Mob projectiles can destroy **crafting tables** and **furnaces** on impact.

## Mob despawn

Mobs despawn when their Manhattan distance from the player exceeds **14**. Exception: during the boss fight on floor 8, all melee and ranged mobs are exempt from despawn so that summoned waves persist until killed. Passive mobs (cow, bat, snail) follow the normal despawn rule.

## Step-loop order

Per turn:
1. Action handler (movement / crafting / DO / etc.)
2. Player projectiles advance (and deal damage)
3. Mob AI runs (each mob takes one step)
4. Mob projectiles advance (and deal damage)
5. Despawn sweep
6. Survival vitals decay / lava damage
7. Milestone checks

This ordering means the player always acts before mobs respond, and projectiles fired this turn may hit targets before mobs have moved.
