# Craftax — Progression

## Experience points (XP)

XP is earned by descending to new floors for the first time:

- **+1 XP** on first DESCEND to each new floor (floors 1 through 8).
- Maximum XP from floor entries: **8 XP** (one per dungeon floor).
- XP is tracked by the `_xp_floors_visited` set; re-visiting a floor does not grant XP.

XP is displayed in the HUD. Unspent XP accumulates.

## Attributes

There are 3 RPG attributes: **DEX** (Dexterity), **STR** (Strength), **INT** (Intelligence).

- Each attribute starts at **1** and caps at **5**.
- Spend 1 XP per level-up via LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH, or LEVEL_UP_INTELLIGENCE.
- Assignments are permanent within the episode.
- Each level-up fires the corresponding achievement (`level_up_dexterity` / `level_up_strength` / `level_up_intelligence`).

## Per-attribute scaling

| Attribute | Effect | Formula |
|---|---|---|
| DEX | Max Food | base 9 + (DEX - 1) × 2 |
| DEX | Max Drink | base 9 + (DEX - 1) × 2 |
| DEX | Max Energy | base 9 + (DEX - 1) × 2 |
| DEX | Need-decay rate | × (1 - 0.125 × (DEX - 1)) |
| DEX | Arrow damage | × (1 + 0.2 × (DEX - 1)) |
| STR | Max HP | base 9 + (STR - 1) |
| STR | Melee physical damage | × (1 + 0.25 × (STR - 1)) |
| INT | Max Mana | base 10 + (INT - 1) × 3 |
| INT | Mana regen rate | × (1 + 0.25 × (INT - 1)) |
| INT | Spell damage | × (1 + 0.05 × (INT - 1)) |

## Attribute cap values

| Attribute | Level | Max Food/Drink/Energy | Max HP | Max Mana | Phys dmg mult | Arrow dmg mult | Spell dmg mult | Decay mult |
|---|---|---|---|---|---|---|---|---|
| DEX | 1 | 9 / 9 / 9 | — | — | — | 1.00× | — | 1.000 |
| DEX | 2 | 11 / 11 / 11 | — | — | — | 1.20× | — | 0.875 |
| DEX | 3 | 13 / 13 / 13 | — | — | — | 1.40× | — | 0.750 |
| DEX | 4 | 15 / 15 / 15 | — | — | — | 1.60× | — | 0.625 |
| DEX | 5 | 17 / 17 / 17 | — | — | — | 1.80× | — | 0.500 |
| STR | 1 | — | 9 | — | 1.00× | — | — | — |
| STR | 2 | — | 10 | — | 1.25× | — | — | — |
| STR | 3 | — | 11 | — | 1.50× | — | — | — |
| STR | 4 | — | 12 | — | 1.75× | — | — | — |
| STR | 5 | — | 13 | — | 2.00× | — | — | — |
| INT | 1 | — | — | 10 | — | — | 1.00× | — |
| INT | 2 | — | — | 13 | — | — | 1.05× | — |
| INT | 3 | — | — | 16 | — | — | 1.10× | — |
| INT | 4 | — | — | 19 | — | — | 1.15× | — |
| INT | 5 | — | — | 22 | — | — | 1.20× | — |

## Recommended strategies

**Early game (floors 0-2)**: DEX is valuable for survival (slower food/drink decay). STR helps with floor 4 knights (50% phys defense, so raw melee is less efficient — use enchants).

**Mid game (floors 3-5)**: INT enables effective spellcasting needed for floors 6 and 7. Prioritize INT once a book is obtained and spells are learned.

**Endgame (floors 6-8)**: Floor 6 mobs have 0.9 phys defense and 1.0 fire immunity — ice spells are mandatory. Floor 7 mobs have 0.9 phys defense and 1.0 ice immunity — fire spells/enchants are mandatory. Boss floor 8 applies 1.5× damage multiplier; higher STR and armor reduce incoming damage significantly.

## XP allocation advice

With 8 XP total available, common splits:
- DEX 3, STR 2, INT 3 = 8 pts: balanced survival + damage.
- DEX 2, STR 2, INT 4 = 8 pts: magic-focused (max INT 5 requires 4 XP on INT alone).
- STR 5, INT 1, DEX 2 = 8 pts: melee focus (only viable with enchanted weapons for floors 6-7).

Note: reaching floor 8 requires surviving through floor 7, which requires fire damage. Pure STR builds will struggle.
