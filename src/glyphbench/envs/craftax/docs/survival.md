# Craftax — Survival

## Vitals overview

The agent has five vitals: HP, Food, Drink (water), Energy, and Mana.

| Vital | Base max | Scales with |
|---|---|---|
| HP | 9 | STR (+1 per point above 1) |
| Food | 9 | DEX (+2 per point above 1) |
| Drink | 9 | DEX (+2 per point above 1) |
| Energy | 9 | DEX (+2 per point above 1) |
| Mana | 10 | INT (+3 per point above 1) |

All vitals start at their maximum at episode start. Caps are re-evaluated at runtime as attributes change.

## Need decay

Food, Drink, and Energy decay on a fixed interval:

| Vital | Drain interval (base) |
|---|---|
| Food | 1 per 50 steps |
| Drink | 1 per 40 steps |
| Energy | 1 per 100 steps |

**DEX scaling**: the decay multiplier is `1.0 - 0.125 × (DEX - 1)`. At DEX 5 the multiplier is 0.5, meaning needs drain at half the base rate.

## Consequences of depletion

- **Food = 0**: -1 HP per step.
- **Drink = 0**: -1 HP per step.
- **Energy = 0**: 50% chance any move action fails.

HP can deplete from multiple sources simultaneously (e.g., food + drink both zero = -2 HP/step).

## Mana regeneration

Mana regenerates passively at 1 per 20 steps (base). The regen rate is multiplied by `1.0 + 0.25 × (INT - 1)`. At INT 5 the regen is 2× the base rate. Mana is used by spells (2 mana each) and enchanting (9 mana). See `magic.md` and `crafting.md`.

## SLEEP state machine

Action: **SLEEP** (action index 6).

1. Entering: sets `_is_sleeping = True`. The agent remains stationary.
2. Per-tick while sleeping:
   - +2 HP (capped at max HP).
   - +2 Energy (capped at max Energy).
3. Exit conditions:
   - Energy reaches max: normal wake (fires `wake_up` achievement).
   - Incoming damage: sleep is cancelled immediately. No `wake_up` achievement.
4. **Vulnerability**: while sleeping, all incoming damage is multiplied by **3.5×** (applied after armor reduction, before boss-floor multiplier). Do not sleep in open areas with mobs present; block yourself in with placed stone first.
5. Combined multiplier on floor 8 (boss floor): sleeping × boss = **3.5 × 1.5 = 5.25×** effective incoming damage.

Food and Drink continue to drain while sleeping.

## REST state machine

Action: **REST** (action index 33).

1. Entering: sets `_is_resting = True`. The agent executes no-op ticks.
2. Per-tick while resting:
   - +1 HP (capped at max HP).
3. Exit conditions (any one suffices):
   - HP reaches max.
   - Food reaches 0.
   - Drink reaches 0.
   - Incoming damage (any source, including mob hits or lava).
4. REST does not accelerate energy recovery. Use SLEEP for energy; use REST for HP top-up when energy is already full.
5. While resting the sleep damage multiplier does **not** apply. Only the 3.5× multiplier applies during `_is_sleeping`.

## Lava damage

Standing on a `♨` (TILE_LAVA) tile deals **2 HP per tick** regardless of armor. Lava is walkable; the damage fires every step the agent occupies the tile. This applies on any floor containing lava (floor 6 Fire Realm is the primary lava floor, but lava may appear in other dungeon generators).

## Day/night cycle (floor 0 only)

The overworld has a day/night cycle: 200-step day, 100-step night (300-step full cycle).

- During the day the lightmap is fully lit on the surface.
- At night the ambient light drops; mob spawn rate scales with `(1 - light_level)²`. At full darkness (light = 0), effective night spawn chance reaches `_NIGHT_SPAWN_BASE_CHANCE` (= 1.0 per step). This makes nighttime significantly more dangerous.
- Dungeon floors do not have day/night cycles; their light comes exclusively from torches and biome baselines. Floor 2 (Gnomish Mines) is dark by default — torch placement is required for visibility.

## Visibility / lightmap

Visibility is computed per-tile from a float lightmap. A cell is visible if its light value exceeds 0.05. Sources of light:

- **Day/night**: surface light from the day counter.
- **Torches**: PLACE_TORCH emits light at radius 5 from the placed tile.
- **Biome baseline**: rooms-and-corridors floors (1, 3, 4) have a baseline light level that makes them partially visible without torches. Smoothgen floors (2, 5, 6, 7, 8) start dark.

See `items.md` for torch crafting details.
