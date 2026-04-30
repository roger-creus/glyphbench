"""Phase-α projectile entity + container tests."""
from glyphbench.envs.craftax.full import CraftaxFullEnv
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
