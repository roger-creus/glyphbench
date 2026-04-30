# Craftax — Magic

## Overview

The agent can learn two spells: **fireball** and **iceball**. Spells must be learned before use. Each spell fires a single-target traveling projectile and costs **2 mana**. Spell damage scales with the INT attribute.

## Spell list

| Spell | Action | Mana cost | Element | Projectile glyph |
|---|---|---|---|---|
| Fireball | CAST_FIREBALL | 2 | Fire | `●` (or `◉` for mob variant) |
| Iceball | CAST_ICEBALL | 2 | Ice | `○` (or `◎` for mob variant) |

Both spells:
- Spawn a projectile at the player's tile facing in the current direction.
- The projectile travels 1 tile per turn.
- Collision is checked at pre-advance and post-advance positions each tick.
- Projectile is consumed on hitting a mob, a solid tile, or leaving bounds.
- Deal damage of the matching element — bypasses mobs immune to the opposite element.

Use fireball against ice-realm mobs (frost_troll, ice_elemental). Use iceball against fire-realm mobs (pigman, fire_elemental). Physical weapons are nearly ineffective against floors 6 and 7.

## Damage scaling

Spell damage (base scalar) is multiplied by `1.0 + 0.05 × (INT - 1)`.

| INT | Multiplier |
|---|---|
| 1 | 1.00× |
| 2 | 1.05× |
| 3 | 1.10× |
| 4 | 1.15× |
| 5 | 1.20× |

The scaled damage is applied as a fire or ice 3-vector `(0, damage, 0)` or `(0, 0, damage)`. Per-mob fire/ice immunity (defense = 1.0) blocks this damage entirely. See `combat.md` for the defense formula.

## Learning spells

Spells are learned via the **READ_BOOK** action:

1. Obtain a book from a chest (see `items.md` for chest loot gating).
2. With the book in inventory, perform READ_BOOK.
3. READ_BOOK consumes 1 book and randomly teaches one **unlearned** spell.
4. If both spells are already known, READ_BOOK has no effect.

The `_learned_spells` dict tracks per-spell flags: `{"fireball": bool, "iceball": bool}`. CAST_FIREBALL is gated on `_learned_spells["fireball"]`; CAST_ICEBALL is gated on `_learned_spells["iceball"]`.

Achievements: `learn_fireball` and `learn_iceball` fire on first learning. `cast_fireball` and `cast_iceball` fire on first cast.

## Mana management

- Mana starts at max (10 base; +3 per INT point above 1).
- Mana regenerates passively: 1 per 20 steps × `(1.0 + 0.25 × (INT - 1))` regen multiplier.
- Enchanting consumes 9 mana. Two spells in rapid succession cost 4 mana total.
- Track mana in the HUD before casting. With INT 1 at base 10 mana, you can cast 5 times before exhausting mana; regen at INT 1 = 1/20 steps means ~100 steps to recharge from empty.
- At INT 5: max mana = 10 + 12 = 22; regen multiplier = 2.0 (1 per 10 steps).

## Books

Books are found in chests with a floor-gated priority:

- Floor 1 first chest: grants **bow** (not book). See `items.md`.
- Floor 3 first chest: grants **book**. Achievement: `find_book`.
- Floor 4 first chest: grants **book** (if floor 3 chest has not yet been opened).

Books are stored in the `book` inventory key. The READ_BOOK action decrements `book` by 1 and teaches a spell.

There is no way to obtain additional books once the first-chest grants are exhausted. If you use READ_BOOK and already know both spells, the book is consumed with no benefit, so check `_learned_spells` before acting.
