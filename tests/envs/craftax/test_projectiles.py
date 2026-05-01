"""Phase-α projectile entity + container tests."""
from glyphbench.envs.craftaxfull.full import CraftaxFullEnv
from glyphbench.envs.craftax.mechanics.projectiles import (
    ProjectileEntity,
    ProjectileType,
    step_player_projectiles,
)


def test_projectile_entity_construction() -> None:
    p = ProjectileEntity(
        kind=ProjectileType.ARROW,
        x=5,
        y=7,
        dx=1,
        dy=0,
        damage=2,
    )
    assert p.kind == ProjectileType.ARROW
    assert (p.x, p.y) == (5, 7)
    assert (p.dx, p.dy) == (1, 0)
    assert p.damage == 2


def test_projectile_type_enum_has_eight_upstream_variants() -> None:
    """Upstream Craftax constants.py:125-134 defines 8 ProjectileType values."""
    expected = {"ARROW", "DAGGER", "FIREBALL", "ICEBALL",
               "ARROW2", "SLIMEBALL", "FIREBALL2", "ICEBALL2"}
    assert {p.name for p in ProjectileType} == expected


def test_projectile_type_enum_integer_values_match_upstream() -> None:
    """Pin integer values — downstream code (T19 RANGED_MOB_TO_PROJECTILE,
    T26 renderer) indexes by .value, so order matters as much as names."""
    assert ProjectileType.ARROW.value == 0
    assert ProjectileType.DAGGER.value == 1
    assert ProjectileType.FIREBALL.value == 2
    assert ProjectileType.ICEBALL.value == 3
    assert ProjectileType.ARROW2.value == 4
    assert ProjectileType.SLIMEBALL.value == 5
    assert ProjectileType.FIREBALL2.value == 6
    assert ProjectileType.ICEBALL2.value == 7


def test_projectile_advances_one_tile_per_step() -> None:
    p = ProjectileEntity(
        kind=ProjectileType.ARROW, x=5, y=7, dx=1, dy=0, damage=2,
    )
    p.advance()
    assert (p.x, p.y) == (6, 7)
    p.advance()
    assert (p.x, p.y) == (7, 7)


def test_full_env_has_projectile_lists_after_reset() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    # New state fields exist and start empty.
    assert env._player_projectiles == []
    assert env._mob_projectiles == []


def test_player_projectile_advances_each_step() -> None:
    proj = ProjectileEntity(kind=ProjectileType.ARROW, x=5, y=5, dx=1, dy=0, damage=2)
    step_player_projectiles([proj], map_w=20, map_h=20, blocked_fn=lambda p: False, hit_fn=lambda p: False)
    assert (proj.x, proj.y) == (6, 5)


def test_player_projectile_expires_off_map() -> None:
    proj = ProjectileEntity(kind=ProjectileType.ARROW, x=19, y=5, dx=1, dy=0, damage=2)
    survivors = step_player_projectiles([proj], map_w=20, map_h=20, blocked_fn=lambda p: False, hit_fn=lambda p: False)
    assert survivors == []


def test_player_projectile_stops_at_solid_block() -> None:
    """A projectile entering a blocked tile is dropped. Its terminal position
    is the blocked tile itself (because step_player_projectiles advances
    BEFORE running the blocked_fn check)."""
    proj = ProjectileEntity(kind=ProjectileType.ARROW, x=5, y=5, dx=1, dy=0, damage=2)
    blocked = {(6, 5)}
    survivors = step_player_projectiles(
        [proj],
        map_w=20, map_h=20,
        blocked_fn=lambda p: (p.x, p.y) in blocked,
        hit_fn=lambda p: False,
    )
    assert survivors == []
    # Terminal position is the blocked tile (proj advanced into it before being dropped).
    assert (proj.x, proj.y) == (6, 5)


def test_player_projectile_damages_one_mob_then_dies() -> None:
    """A projectile that hits a mob (hit_fn returns True) is dropped from
    the survivors list. Subsequent advances along the same path are not
    queried, so the projectile cannot 'pass through' to a second mob.

    NB: T07's step_player_projectiles only advances projectiles by ONE tile
    per call. Per-step semantics are tested over a SINGLE call, so a single
    projectile hitting one mob in its path must produce exactly one hit_fn
    invocation at the post-advance tile.
    """
    proj = ProjectileEntity(kind=ProjectileType.ARROW, x=5, y=5, dx=1, dy=0, damage=2)
    mob_hits: list[tuple[int, int, int]] = []  # (x, y, damage)

    def hit_and_damage(p) -> bool:
        # Tile (6,5) holds a mob.
        if (p.x, p.y) == (6, 5):
            mob_hits.append((p.x, p.y, p.damage))
            return True  # absorb the projectile
        return False

    survivors = step_player_projectiles(
        [proj], map_w=20, map_h=20,
        blocked_fn=lambda p: False,
        hit_fn=hit_and_damage,
    )
    assert survivors == []
    # Exactly one hit was registered.
    assert mob_hits == [(6, 5, 2)]


def test_render_shows_arrow_projectile_glyph() -> None:
    """Phase α: arrow projectiles render as their assigned glyph."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = 5, 5
    env._player_projectiles.append(ProjectileEntity(
        kind=ProjectileType.ARROW, x=7, y=5, dx=1, dy=0, damage=2,
    ))
    obs = env._render_current_observation()
    grid_text = obs.grid
    # Check that an arrow projectile glyph appears somewhere in the rendered window.
    # The glyph is whatever T26 picks — assert against the constant.
    from glyphbench.envs.craftax.base import TILE_ARROW
    assert TILE_ARROW in grid_text, f"arrow glyph not in render:\n{grid_text}"


def test_render_shows_fireball_projectile_glyph() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = 5, 5
    env._player_projectiles.append(ProjectileEntity(
        kind=ProjectileType.FIREBALL, x=7, y=5, dx=1, dy=0, damage=4,
    ))
    obs = env._render_current_observation()
    grid_text = obs.grid
    from glyphbench.envs.craftax.base import TILE_FIREBALL
    assert TILE_FIREBALL in grid_text, f"fireball glyph not in render:\n{grid_text}"


def test_projectile_glyphs_unique_within_palette() -> None:
    """Phase α: each ProjectileType maps to a unique glyph (single codepoint)."""
    from glyphbench.envs.craftax.base import (
        TILE_ARROW, TILE_DAGGER, TILE_FIREBALL, TILE_ICEBALL,
        TILE_ARROW2, TILE_SLIMEBALL, TILE_FIREBALL2, TILE_ICEBALL2,
    )
    glyphs = (
        TILE_ARROW, TILE_DAGGER, TILE_FIREBALL, TILE_ICEBALL,
        TILE_ARROW2, TILE_SLIMEBALL, TILE_FIREBALL2, TILE_ICEBALL2,
    )
    # Each is a single codepoint.
    for g in glyphs:
        assert len(g) == 1, f"glyph {g!r} is not single-codepoint"
    # All distinct.
    assert len(set(glyphs)) == len(glyphs), f"duplicate glyphs: {glyphs}"


def test_projectile_advances_diagonally_on_both_axes() -> None:
    """T_FOLLOWUP_D: per-call advance increments both x and y when (dx, dy)
    is non-cardinal (e.g., (1, 1) — diagonal). Phase α projectiles only fire
    cardinally in practice, but the entity API supports diagonal motion."""
    p = ProjectileEntity(kind=ProjectileType.ARROW, x=5, y=5, dx=1, dy=1, damage=2)
    p.advance()
    assert (p.x, p.y) == (6, 6)
    p.advance()
    assert (p.x, p.y) == (7, 7)


def test_player_projectile_blocks_on_plant_tile() -> None:
    """T_FOLLOWUP_B: upstream SOLID_BLOCKS (constants.py:370-371) includes
    BlockType.PLANT and RIPE_PLANT. Projectiles must stop on plant tiles."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = 5, 5
    env._facing = (1, 0)
    env._mana = 5
    env._learned_spells["fireball"] = True
    grid = env._current_grid()
    # Place a ripe plant at (7, 5) — projectile from (5, 5) east.
    from glyphbench.envs.craftax.base import TILE_RIPE_PLANT
    grid[5][7] = TILE_RIPE_PLANT

    # Cast fireball; it should travel to (6, 5) on the cast step, then on
    # the next NOOP advance to (7, 5) and be BLOCKED (dropped) by the plant.
    env.step(env.action_spec.names.index("CAST_FIREBALL"))
    env.step(env.action_spec.names.index("NOOP"))
    # No surviving fireball.
    assert env._player_projectiles == [], (
        f"projectile should be blocked on ripe-plant tile; survivors={env._player_projectiles}"
    )


# ---------------------------------------------------------------------------
# T19γ: Pre-advance projectile collision (T_FOLLOWUP_C)
# ---------------------------------------------------------------------------

def test_player_projectile_pre_advance_hit_consumes_projectile() -> None:
    """Phase γ T19γ: a mob at the projectile's SPAWN tile (same coords) is hit
    BEFORE the advance step. The projectile must be consumed immediately.

    hit_fn fires on the pre-advance position (5, 5). The projectile never
    advances to (6, 5). survivors must be empty.
    """
    proj = ProjectileEntity(kind=ProjectileType.ARROW, x=5, y=5, dx=1, dy=0, damage=3)
    hits: list[tuple[int, int]] = []

    def hit_fn(p) -> bool:
        # Mob is stationary at (5, 5) — the projectile's initial tile.
        if (p.x, p.y) == (5, 5):
            hits.append((p.x, p.y))
            return True
        return False

    survivors = step_player_projectiles(
        [proj], map_w=20, map_h=20,
        blocked_fn=lambda p: False,
        hit_fn=hit_fn,
    )
    # Projectile consumed pre-advance.
    assert survivors == [], f"projectile should be consumed pre-advance; survivors={survivors}"
    # The hit occurred exactly once at the pre-advance tile.
    assert hits == [(5, 5)], f"expected pre-advance hit at (5,5); got {hits}"
    # The projectile must NOT have advanced (position unchanged from spawn).
    assert (proj.x, proj.y) == (5, 5), (
        f"projectile should not have moved; pos=({proj.x},{proj.y})"
    )


def test_player_projectile_pre_advance_hit_no_post_advance_hit() -> None:
    """Phase γ T19γ: after a pre-advance hit the projectile is consumed and
    does NOT travel further. A second mob at (6, 5) must NOT be struck.
    """
    proj = ProjectileEntity(kind=ProjectileType.ARROW, x=5, y=5, dx=1, dy=0, damage=3)
    hit_positions: list[tuple[int, int]] = []

    def hit_fn(p) -> bool:
        hit_positions.append((p.x, p.y))
        # Mob A at (5, 5) absorbs the shot.
        if (p.x, p.y) == (5, 5):
            return True
        return False

    survivors = step_player_projectiles(
        [proj], map_w=20, map_h=20,
        blocked_fn=lambda p: False,
        hit_fn=hit_fn,
    )
    assert survivors == []
    # hit_fn must have been called exactly once (pre-advance only, not again
    # at (6,5) because the projectile was consumed before advancing).
    assert len(hit_positions) == 1
    assert hit_positions[0] == (5, 5)
