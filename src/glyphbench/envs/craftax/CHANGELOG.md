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

## Phase β — Survival, Magic & Loot (2026-04-29)

### Added
- `mechanics/potions.py` — 6-color potion shuffle (`make_potion_mapping`,
  `apply_potion_effect`, `POTION_COLORS`, `POTION_EFFECTS`).
- `mechanics/lighting.py` — per-tile lightmap subsystem (`compute_lightmap`,
  `TORCH_RADIUS=5`, `VISIBILITY_THRESHOLD=0.05`).
- `mechanics/world_gen.py` — dungeon-room biome generator with 8 rooms,
  L-shaped corridors, chests, fountains.
- New full-spec actions: `REST` (35), `READ_BOOK` (42), `DRINK_POTION_RED/
  GREEN/BLUE/PINK/CYAN/YELLOW` (36-41). Net spec change: -1 generic
  `DRINK_POTION` (legacy, removed) + 8 new = +7. Spec post-β: 36 (post-α)
  + 7 = 43.
- New tile constants: `TILE_SAPPHIRE` (♦), `TILE_RUBY` (▼), `TILE_CHEST`
  ($), `TILE_FOUNTAIN` (⊙), `TILE_ENCHANT_FIRE` (Ⓔ), `TILE_ENCHANT_ICE`
  (Ⓘ).
- New inventory keys: `sapphire`, `ruby`, `book`, `potions[red/green/blue/
  pink/cyan/yellow]`. The legacy `_potions: list[str]` is retained as a
  no-op field for backward compat; the per-color `inventory["potions"]`
  dict is the live state.
- New env state: `_is_resting: bool`, `_potion_mapping: tuple[int, ...]`
  (per-game hidden shuffle), `_learned_spells: dict[str, bool]`,
  `_chests_opened: dict[int, set]`, `_first_chest_opened: dict[int, bool]`,
  `_lightmap: dict[int, np.ndarray]`, `_achievements_phase_beta: dict[str,
  bool]` (67-key bitmap from upstream constants.py:406-585).

### Changed
- `SLEEP` (action 6): no longer an instant 49-step skip with random shelter
  ambush. Now enters a continuous `_is_sleeping` state with +2 HP/tick and
  +2 energy/tick regen. Exits on `_energy >= _MAX_ENERGY` (fires `wake_up`)
  or on incoming damage (no achievement). Sleep is interruptible by mobs.
- `_take_damage`: cancels both `_is_sleeping` and `_is_resting` on any hit.
- Sapphire/ruby ore on floor 2 (Gnomish Mines): 2.5% each on stone-eligible
  tiles, mutually exclusive with diamond/iron/coal. Iron-pickaxe gating
  enforced via `PICKAXE_REQUIRED` table (tier 2).
- Lava: previously walk-blocked or fire-resist gated. Now walkable; player
  takes 2 damage per tick while standing on `TILE_LAVA`. The legacy
  `_fire_resist_turns` field is dropped entirely.
- Chest interaction: DO on `TILE_CHEST` rolls upstream loot table (wood /
  torches / ore / potion / arrows / pickaxe / sword) per
  `add_items_from_chest`. Each chest opens once per episode.
- First-chest gating: floor 1 first chest grants 1 bow + `find_bow`
  achievement. First chest on floor 3 OR 4 grants 1 book + `find_book`.
- Spell tracking: legacy `_spells_learned: int` counter replaced by
  `_learned_spells: dict[str, bool]` per-spell. `CAST_FIREBALL` /
  `CAST_ICEBALL` gate on their respective bool.
- Visibility: `_is_visible(x, y)` now reads `_lightmap[floor][y, x] > 0.05`
  instead of binary 3-tile-radius + 4-tile-torch-radius. Torches, day/night
  cycle, and biome baselines all feed the lightmap.
- Spawn rate: melee night spawns scale by `(1 - light_level)²` per the
  upstream darkness² rule.
- Floor 1, 3, 4 generation: rewritten via `generate_dungeon_floor` — 8 rooms
  with L-shaped corridors, 1 chest per room, ~50% chance fountain per room.
  Floor 3 hosts `TILE_ENCHANT_ICE`; floor 4 hosts `TILE_ENCHANT_FIRE`
  (enchantment semantics deferred to phase γ).

### Removed
- Legacy single-action `DRINK_POTION` (replaced by 6 per-color actions).
- Legacy achievements `drink_health_potion`, `drink_fire_resist_potion`,
  `drink_speed_potion` (consolidated into `drink_potion`).
- State field `_fire_resist_turns` (no fire-resist potion in upstream).
- Legacy mob names `skeleton_archer` and `spider`. Renamed to upstream-
  faithful `skeleton` (ranged) and `kobold` (ranged). Achievement
  `defeat_skeleton_archer` renamed to `defeat_skeleton`; `defeat_spider`
  renamed to `defeat_kobold`. (T_FOLLOWUP_A complete.)

### Action spec post-β
43 actions total: 36 (post-α) + 1 `REST` + 6 `DRINK_POTION_*` + 1
`READ_BOOK` = 43. Phase γ adds 4 more (LEVEL_UP_DEX/STR/INT, ENCHANT_BOW),
making the projected final spec 47.

### Test count
Phase α end: 146 craftax tests. Phase β end: 275 craftax tests.

### Deferred (still tracked)
- T_FOLLOWUP_C: pre-advance projectile collision (phase γ).
- T_FOLLOWUP_E: ranged-mob cornered fallback (phase γ).
- Sapphire/ruby ore on floors 4-7 (phase γ, pending floor generation).
- Re-baseline (random + LLM smoke) — deferred for cluster_manager runs.

## Phase γ — Progression & Endgame (2026-04-29)

### Added
- `mechanics/damage.py` — 3-vector damage `(phys, fire, ice)`,
  `damage_vec_from_projectile`, `player_defense_vec`,
  `damage_dealt_to_player`, `damage_dealt_to_mob`. Multiplicative armour
  formula: 0.1/slot phys + 0.2/slot matching enchant element.
- `mechanics/progression.py` — per-attribute scaling helpers
  (max_hp_from_str, max_food_from_dex, damage_scale_phys,
  damage_scale_arrow, damage_scale_spell, mana_regen_scale, decay_scale).
- `mechanics/boss.py` — necromancer state machine (is_necromancer_vulnerable,
  damage_necromancer_if_vulnerable, boss_progress_win).
- New full-spec actions (4): `LEVEL_UP_DEXTERITY/STRENGTH/INTELLIGENCE`
  + `ENCHANT_BOW`. Spec post-γ: 45 actions.
- New tile constants: `TILE_FIRE_TREE` (♠), `TILE_ICE_SHRUB` (❄),
  `TILE_GRAVE` (⚰), `TILE_NECROMANCER` (Ⓝ),
  `TILE_NECROMANCER_VULNERABLE` (ⓝ), `TILE_TROLL`, `TILE_DEEP_THING`,
  `TILE_SNAIL`, `TILE_PIGMAN`, `TILE_FIRE_ELEMENTAL`, `TILE_FROST_TROLL`,
  `TILE_ICE_ELEMENTAL`.
- New env state fields: `_armor_slots: dict[str, int]` (helmet/chest/
  legs/boots → tier), `_armor_enchants: dict[str, int]` (slot → element
  0/1/2), `_xp: int`, `_xp_floors_visited: set[int]`, `_dex/_str/_int_attr:
  int (1..5 cap)`, `_sword_enchantment: int (0/1/2)`, `_bow_enchantment:
  int (0/1/2)`, `_boss_progress: int`, `_boss_summon_timer: int`. Also
  `_max_food/drink/energy/mana` per-instance fields (replacing module
  constants) so attribute-scaled maxes can change at runtime.
- Floors 5/6/7/8 generation: Troll Mines (smoothgen), Fire Realm
  (smoothgen, lava + ruby + fire trees), Ice Realm (smoothgen, water +
  sapphire + ice shrubs), Graveyard (smoothgen, graves + necromancer).
- 12 new mob tiles for the new floors' rosters (troll, deep_thing,
  pigman, fire_elemental, frost_troll, ice_elemental, snail).
- 3 new dungeon-progression achievements: `enter_fire_realm`,
  `enter_ice_realm`, `enter_graveyard` (T22γ). `ALL_FULL_ACHIEVEMENTS`
  total: 90 → 93.
- HUD now shows `Upstream achievements: X / 67` (bitmap-based count from
  `_achievements_phase_beta`, faithfully mirrors the 67 upstream enum
  entries). `_try_unlock` now also flips the bitmap on every unlock (T21γ).

### Changed
- `_take_damage(scalar)`: now accepts an optional 3-vec; legacy scalar
  treated as physical. Routes through multiplicative armour reduction.
  Boss-floor 1.5× multiplier applied AFTER armour. Sleep multiplier
  3.5× applied AFTER boss multiplier (5.25× max for sleeping on boss
  floor).
- `MAKE_IRON_ARMOUR` / `MAKE_DIAMOND_ARMOUR`: now place into the lowest
  empty slot rather than the legacy single-armour-slot.
- `ENCHANT_SWORD`: requires adjacency to FIRE table (floor 4) OR ICE
  table (floor 3); 1 ruby OR 1 sapphire; 9 mana. Sets element-specific
  enchantment (0/1/2).
- `ENCHANT_ARMOUR`: same prerequisites as sword; per-slot element
  enchant on the lowest unenchanted armoured slot.
- Mob AI: per-mob defense vector applied — fire elementals immune to
  fire damage; ice elementals immune to ice damage; vault knights at
  half phys defense; troll/deep_thing at 0.2 phys defense.
- `step_player_projectiles` / `step_mob_projectiles`: now check collision
  at BOTH pre- and post-advance positions per upstream
  game_logic.py:1786-1799 (resolves T_FOLLOWUP_C).
- `step_ranged_mob`: cornered fallback — when blocked in all 4 cardinals
  AND too close, force shoot (resolves T_FOLLOWUP_E).
- Necromancer state machine on floor 8: vulnerable iff no mobs on floor
  AND summon timer expired; each hit triggers 7-turn summon wave; 8 hits
  to win.
- `_NUM_DUNGEON_FLOORS` 5 → 8 (final 9 floors total).
- `_is_in_boss_fight()` now keys on floor 8 + necromancer alive (not
  legacy floor 5).
- System prompt (T23γ): full phase γ BACKGROUND/COMBAT/ATTRIBUTES/MAGIC/
  ENCHANTMENTS/POTIONS/BOSS/DUNGEONS sections with 9-floor map, 4 armor
  slots, 3 attributes (DEX/STR/INT), necromancer mechanics.

### Removed
- Action `MAKE_WOOD_ARMOR` (legacy; not in upstream — armour is iron+
  only).
- Action `MAKE_STONE_ARMOR` (legacy).
- State field `_weapon_enchanted: bool` (replaced by
  `_sword_enchantment: int`).
- Inventory keys `wood_armor`, `stone_armor`, `iron_armor`,
  `diamond_armor` (replaced by `_armor_slots` per-slot dict).
- Achievement `enchant_weapon` (renamed to `enchant_sword`).

### Action spec post-γ
45 actions. Phase α removed 2 + added 3 (net +1 → 36); phase β added
1 REST + 6 DRINK_POTION_<color> + 1 READ_BOOK − 1 generic DRINK_POTION
(net +7 → 43); phase γ removed 2 (MAKE_WOOD_ARMOR, MAKE_STONE_ARMOR)
+ added 4 (LEVEL_UP_DEX/STR/INT + ENCHANT_BOW) (net +2 → 45).

### Test count
Phase β end: 275 craftax tests. Phase γ end: ~375+ craftax tests.

### Resolved follow-ups
- T_FOLLOWUP_A: legacy mob name alignment (resolved in phase β T04).
- T_FOLLOWUP_B: PLANT/RIPE_PLANT in _SOLID_TILES (resolved in phase α
  T31).
- T_FOLLOWUP_C: pre-advance projectile collision (resolved in T19γ).
- T_FOLLOWUP_D: diagonal-advance projectile test (resolved in phase α
  T31).
- T_FOLLOWUP_E: ranged-mob cornered fallback (resolved in T20γ).

### Next: T_FINAL — LLM-first tutorial markdown docs
Author standardised tutorial in `src/glyphbench/envs/craftax/docs/`
covering all 9 floors, mechanics, and per-task subset prompts. See
memory `project_craftax_tutorial.md`.

## T_FINAL — LLM-first tutorial docs (planned)
After phase γ completes: standardised markdown tutorials in
`src/glyphbench/envs/craftax/docs/` covering all floors, mechanics,
and per-task subset prompts. See memory `project_craftax_tutorial.md`
for the full deliverable spec.
