# Craftax port — changelog

This file tracks behavioural deltas to the Craftax envs as the
multi-phase upstream-faithful port (phases α / β / γ) lands.

## Phase α — Combat & Mobs (2026-04-29)

### Added
- `mechanics/projectiles.py` — `ProjectileEntity` dataclass, `ProjectileType`
  enum (8 variants matching upstream `constants.py:125-134`),
  `step_player_projectiles` (per-tick driver), `step_mob_projectiles` (mob
  side, with table/furnace destruction).
- `mechanics/mobs.py` — `MobType` enum (4 classes: PASSIVE, MELEE, RANGED,
  PROJECTILE), 24-mob roster (3 passive + 8 melee + 8 ranged matching
  upstream), `FLOOR_MOB_MAPPING` (9 floors), `RANGED_MOB_TO_PROJECTILE`,
  `step_melee_mob` (cooldown=5 + 75%-chase / random-walk + boss-fight
  always-chase), `step_ranged_mob` (kiting AI + cooldown=4 + projectile
  spawn), `should_despawn` (dist >= 14 with boss-fight exemption).
- New full-spec actions: `SHOOT_ARROW` (24), `MAKE_ARROW` (25), `MAKE_TORCH`
  (38). Indices match upstream where the upstream value is stable.
- New inventory keys: `bow`, `arrows`, `torch`.
- New env state: `_player_projectiles: list[ProjectileEntity]`,
  `_mob_projectiles: list[ProjectileEntity]`,
  `_pending_step_reward: float`, `_is_sleeping: bool`.
- New per-mob field on `Mob` TypedDict: `attack_cooldown: int`.
- New env helpers: `_is_in_bounds`, `_tile_at`, `_is_in_boss_fight`,
  `_step_player_projectiles`, `_step_mob_projectiles`.
- New module-level constant: `_SOLID_TILES` (frozenset of
  projectile-blocking tiles, including PLANT/RIPE_PLANT per upstream).
- 8 new projectile glyphs in the renderer: `↗ ↘ † ● ◉ ○ ◎ ◐` (single
  codepoint, disjoint from existing palette).

### Changed
- `CAST_FIREBALL` (action 26): no longer 2-tile-radius AOE for 3 mana.
  Now spawns a 1-tile/turn point projectile for 2 mana, mirroring upstream
  `cast_spell:2547-2599`. Spawn position is the player tile (the per-tick
  driver advances it 1 tile/turn after the cast).
- `CAST_ICEBALL` (action 27): no longer freezes the adjacent mob. Now
  spawns a 1-tile/turn point projectile for 2 mana. Phase α uses scalar
  damage; phase γ promotes to a `(physical, fire, ice)` 3-vector that
  bypasses ice defense per upstream.
- `PLACE_TORCH` (action 28): no longer consumes 1 wood + 1 coal directly.
  Now consumes 1 crafted torch from `inventory.torch`. Torches are crafted
  via `MAKE_TORCH`.
- Melee mob AI (`zombie`, `skeleton`): rewritten via `step_melee_mob`.
  Mobs now respect a 5-tick attack cooldown; players take 3.5x damage
  while sleeping (matching upstream's `1 + 2.5 * is_sleeping`).
- Ranged mob AI (`skeleton_archer`): rewritten via `step_ranged_mob`. Mobs
  now kite the player (advance >= 6, retreat <= 3, shoot at 4-5) with a
  4-tick cooldown. Spawned projectiles are real entities that travel and
  damage the player on hit; previously `skeleton_archer` did instant
  damage at distance <= 4.
- Mob projectiles destroy crafting tables and furnaces on impact.
- Mob projectile hits cancel `_is_sleeping`.
- Step-loop order: handler -> player projectiles advance -> mob AI ->
  mob projectiles advance -> despawn sweep -> milestones.

### Removed
- Action `CAST_HEAL` (was a glyphbench fabrication; not in upstream).
  Healing is potion-only upstream; phase β implements the 6-color potion
  shuffle.
- Action `MAKE_SPELL_SCROLL` (was a glyphbench fabrication; not in
  upstream). Spells are learned via `READ_BOOK` upstream; phase β
  implements books-from-chests.
- `frozen_turns: int` field on the `Mob` TypedDict and the freeze-skip
  guard in the mob-update loop. Iceball does not freeze upstream.
- Achievements `cast_heal` and `make_spell_scroll` (unreachable after
  the action removals above).

### Deferred (tracked as follow-up tasks)
- **T_FOLLOWUP_A**: legacy mob names (`skeleton_archer`, `spider`, etc.)
  diverge from the upstream-faithful T18 roster (`knight_archer`,
  `kobold`, etc.). Pending alignment task.
- **T_FOLLOWUP_C**: upstream's pre-advance projectile collision check
  (game_logic.py:1786-1799) is not implemented. Phase α only checks at
  the post-advance position. Coordinated rewrite in phase γ when
  projectile damage becomes 3-vector.
- **T_FOLLOWUP_E**: ranged mob cornered-fallback (force shoot when
  blocked AND too close) deferred to phase β/γ.

### Action spec size
Net: -2 (removed `CAST_HEAL`, `MAKE_SPELL_SCROLL`) + 3 (added
`SHOOT_ARROW`, `MAKE_ARROW`, `MAKE_TORCH`) = +1.
Pre-α: 35 names. Post-α: 36 names.

### Test count
Pre-α: ~80 craftax tests. Post-α: 146 craftax tests (118 unit/integration
tests added across phase α).

## Phase β — Survival, Magic & Loot (planned)
Deferred mechanics: REST action + new sleep state machine, READ_BOOK +
per-spell `learned_spells[i]` flags, books from chests on floors 3-4,
6-color potion shuffle with random per-game `potion_mapping`, lava
damage on contact, full per-tile lightmap, darkness²-modulated mob
spawn, dungeon-room biome generators for floors 1/3/4, sapphire/ruby
ore on floors 2/4/5/6/7.

## Phase γ — Progression & Endgame (planned)
Deferred mechanics: 3-vector damage `(phys, fire, ice)` everywhere,
multiplicative armor with 4-slot per-element enchant, full enchant
table mechanics (FIRE / ICE) including `ENCHANT_BOW`, XP + 3 attributes
(`LEVEL_UP_DEXTERITY/STRENGTH/INTELLIGENCE`) with cap=5 and per-attr
scaling, floors 4-7 generation (vaults / troll mines / fire realm /
ice realm), full necromancer state machine on floor 8.

## T_FINAL — LLM-first tutorial docs (planned)
After phase γ completes: standardised markdown tutorials in
`src/glyphbench/envs/craftax/docs/` covering all floors, mechanics,
and per-task subset prompts. See memory `project_craftax_tutorial.md`
for the full deliverable spec.
