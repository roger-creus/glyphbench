"""Phase-α projectile entity + container tests."""
from glyphbench.envs.craftax.mechanics.projectiles import (
    ProjectileEntity,
    ProjectileType,
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
