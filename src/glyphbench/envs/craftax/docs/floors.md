# Craftax — Floors

## Floor navigation

- **DESCEND** (action: stand on `⇣` stair-down tile): travel to the next floor.
- **ASCEND** (action: stand on `⇡` stair-up tile): return to the previous floor.
- First DESCEND to a floor grants **+1 XP** (see `progression.md`).
- Achievements fire on first entry: `enter_dungeon`, `reach_floor_2`, etc.
- Each dungeon floor has a stair-up tile to return to the floor above, except floor 8 (no exit).

---

## Floor 0: Overworld

- **Biome**: smoothgen, 64×64.
- **Light**: full day/night cycle (200-step day, 100-step night). Night spawns scale with `(1 - light)²`.
- **Mob roster**: cow (`c`, passive), zombie (`z`, melee), skeleton (`a`, ranged).
- **Special tiles**: STAIRS_DOWN (`⇣`) to floor 1.
- **Resources**: grass, trees (`♣`), stone (`S`), coal (`C`), iron (`I`), water (`≈`), sand (`░`).
- **Strategy**: Mine trees first; craft table → wood pickaxe → stone pickaxe → iron pickaxe. Collect coal for torches. Eat cows (DO while facing) to restore food. Drink from water tiles. Sleep only after blocking yourself in with placed stone — sleeping in the open with zombies nearby is fatal.
- **Notes**: Overworld is the only floor with diamonds on the surface. It is also the only floor where saplings can be grown for renewable food. No sapphire or ruby here.

---

## Floor 1: Dungeon

- **Biome**: dungeon-rooms (32×32), 8 rooms connected by L-shaped corridors.
- **Light**: rooms have a baseline light level; corridors are darker. Torches extend visibility.
- **Mob roster**: snail (`s`, passive, edible), orc_soldier (melee), orc_mage (ranged, fires fireballs).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 0; STAIRS_DOWN (`⇣`) to floor 2; chests (`$`); fountains (`⊙`).
- **First-chest grant**: the first chest opened on floor 1 gives 1 **bow** + `find_bow` achievement. Other chests give standard loot.
- **Fountains**: refill water when interacted with (DO while facing `⊙`).
- **REST usage**: if on low HP, block yourself in and REST for HP recovery without expending energy.
- **Notes**: Orc mages fire fireball projectiles; they can be kited. Collect wood from the surface before descending — wood is scarce underground.

---

## Floor 2: Gnomish Mines

- **Biome**: smoothgen (cave-like), 32×32.
- **Light**: **dark floor** — no ambient light. Place torches to see.
- **Mob roster**: bat (`b`, passive, edible), gnome_warrior (melee), gnome_archer (ranged, fires arrows).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 1; STAIRS_DOWN (`⇣`) to floor 3.
- **Resources**: coal, iron, diamond, sapphire (`♦` at 2.5%), ruby (`▲` at 2.5%). Sapphire/ruby placement is mutually exclusive with diamond/iron/coal on eligible tiles.
- **Ore gating**: sapphire and ruby require iron pickaxe (tier 2) to mine.
- **Strategy**: craft iron pickaxe before descending here to mine sapphire and ruby. Stock torches (MAKE_TORCH) before entering. Mine cave edges for richer ore deposits.
- **Notes**: Gnomes are stronger than floor 1 mobs. Avoid being surrounded in open smoothgen spaces. Bat meat restores food.

---

## Floor 3: Sewers

- **Biome**: dungeon-rooms, 32×32.
- **Light**: rooms have baseline light; place torches for full coverage.
- **Mob roster**: snail (`s`, passive), lizard (melee), kobold (`q`, ranged, fires daggers).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 2; STAIRS_DOWN (`⇣`) to floor 4; chests; fountains; **ICE enchantment table** (`Ⓘ`).
- **First-chest grant**: the first chest opened on floor 3 gives 1 **book** + `find_book` achievement.
- **Enchanting**: stand adjacent to `Ⓘ` with 1 sapphire + 9 mana to perform ice enchantments (ENCHANT_WEAPON, ENCHANT_ARMOR, ENCHANT_BOW).
- **Notes**: Kobolds throw daggers (physical projectiles). Lizards are strong melee mobs. The ice enchant table is critical for floors 7 frost_troll/ice_elemental preparation.

---

## Floor 4: Vaults

- **Biome**: dungeon-rooms, 32×32.
- **Light**: rooms have baseline light.
- **Mob roster**: snail (`s`, passive), knight (melee, 0.5 phys defense), knight_archer (ranged, 0.5 phys defense, fires arrow2 projectiles).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 3; STAIRS_DOWN (`⇣`) to floor 5; chests; fountains; **FIRE enchantment table** (`Ⓔ`).
- **First-chest grant**: if floor 3 first chest was not opened, the floor 4 first chest gives 1 **book**.
- **Enchanting**: stand adjacent to `Ⓔ` with 1 ruby + 9 mana to perform fire enchantments.
- **Knight defense**: both knight and knight_archer have 50% physical defense. Physical-only builds deal half damage. Use enchanted weapons, spells, or enchanted bow.
- **Notes**: Stock ruby before reaching floor 4 to use the fire enchant table immediately. Vault knights are tough; prioritize ranged combat or spells.

---

## Floor 5: Troll Mines

- **Biome**: smoothgen (dark cave), 32×32.
- **Light**: **dark floor** — no ambient light. Place torches.
- **Mob roster**: bat (`b`, passive), troll (`T`, melee, 0.2 phys defense, 12 HP, 4 damage), deep_thing (`d`, ranged, 0.2 phys defense, 8 HP, 3 damage, fires slimeballs).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 4; STAIRS_DOWN (`⇣`) to floor 6.
- **Resources**: sapphire (`♦`), ruby (`▲`) at low density (~1% each). Best floor to accumulate ore for full diamond armor and enchanting.
- **Strategy**: craft full diamond armor here before proceeding. Stock both sapphire (ice enchant) and ruby (fire enchant). Deep things fire slimeball projectiles (mixed phys/fire/ice damage — armor helps but all elements pass through partially).
- **Notes**: Trolls are slow but hit hard. Deep things kite and fire from range 4-5. Bring plenty of arrows and food. Bat meat available for food.

---

## Floor 6: Fire Realm

- **Biome**: smoothgen with lava patches, 32×32.
- **Light**: partially lit by lava ambient (lava tiles); still relatively dark.
- **Mob roster**: bat (`b`, passive), pigman (`p`, melee, 0.9 phys + 1.0 fire immune, 14 HP, 5 damage), fire_elemental (`F`, ranged, 0.9 phys + 1.0 fire immune, 10 HP, 4 damage, fires fireball2).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 5; STAIRS_DOWN (`⇣`) to floor 7; fire trees (`♠`, decoration); lava (`♨`).
- **Resources**: ruby (`▲` at 2.5%), coal; no water on this floor.
- **Lava**: walking on lava deals **2 HP per tick**. Build stone bridges (PLACE_STONE + mine) to cross. No water to drink; return to floor 5 periodically for water.
- **Immunity**: fire-realm mobs are immune to fire damage and 90% resistant to physical. **Iceball + ice-enchanted weapons are mandatory here.** Sapphire enchants must be prepared in advance (floor 3 or 5 enchant access).
- **Notes**: No food source on this floor except bat meat. Stockpile food before entering. Fire elementals fire fireball2 projectiles.

---

## Floor 7: Ice Realm

- **Biome**: smoothgen with water pools, 32×32.
- **Light**: **dark floor** — no ambient light. Place torches.
- **Mob roster**: bat (`b`, passive), frost_troll (`r`, melee, 0.9 phys + 1.0 ice immune, 14 HP, 5 damage), ice_elemental (`i`, ranged, 0.9 phys + 1.0 ice immune, 10 HP, 4 damage, fires iceball2).
- **Special tiles**: STAIRS_UP (`⇡`) to floor 6; STAIRS_DOWN (`⇣`) to floor 8; ice shrubs (`❄`, decoration); water (`≈`).
- **Resources**: sapphire (`▲` at 2%); some ruby; no food source.
- **Immunity**: ice-realm mobs are immune to ice damage and 90% resistant to physical. **Fireball + fire-enchanted weapons are mandatory here.** Ruby enchants must be prepared in advance.
- **Water**: water tiles are walkable and available for drinking (DRINK_WATER while facing `≈`).
- **Notes**: No food — return to floor 6 to eat bat meat if needed. Ice elementals fire iceball2 projectiles. This is the last floor before the boss.

---

## Floor 8: Graveyard (Boss Floor)

- **Biome**: smoothgen, 32×32.
- **Light**: dark floor.
- **Special mob**: Necromancer (`N` / `n` vulnerable). This is the win-condition target.
- **Mob roster during boss fight**: wave-summoned zombies (`z`) and skeletons (`a`) spawn near the player after each Necromancer hit.
- **Special tiles**: grave markers (`⚰`, decoration); no STAIRS_DOWN; STAIRS_UP would be here but floor 8 has no exit — this is the final floor.
- **Survival**: food, drink, and energy do **not** decay on floor 8 (upstream rule). Vitals are frozen for the boss encounter.
- **Boss-floor damage multiplier**: all incoming damage is multiplied **1.5×** (after armor, before sleep multiplier).
- **Strategy**: arrive with full armor (ideally diamond), enchanted weapons, and learned spells. Use fire and ice spells for the summoned waves. See `boss.md` for the Necromancer state machine.
- **Notes**: There is no loot on floor 8 and no way to leave. Ensure you are fully prepared before descending from floor 7.
