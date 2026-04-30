"""Phase-γ T01+T02 tests: 3-vector damage scaffold + per-mob defense vector."""
from __future__ import annotations

import pytest

from glyphbench.envs.craftax.mechanics.damage import damage_vec_from_projectile, vec_sum
from glyphbench.envs.craftax.mechanics.mobs import MOB_DEFENSE_VEC, damage_dealt_to_mob
from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType


def test_arrow_damage_vec():
    assert damage_vec_from_projectile(ProjectileType.ARROW, 2) == (2.0, 0.0, 0.0)


def test_fireball_damage_vec():
    assert damage_vec_from_projectile(ProjectileType.FIREBALL, 4) == (0.0, 4.0, 0.0)


def test_slimeball_damage_vec():
    vec = damage_vec_from_projectile(ProjectileType.SLIMEBALL, 6)
    assert len(vec) == 3
    assert all(abs(c - 2.0) < 1e-9 for c in vec)


def test_vec_sum():
    assert vec_sum((2.0, 0.0, 0.0)) == 2


def test_fire_elemental_defense():
    assert MOB_DEFENSE_VEC["fire_elemental"] == (0.9, 1.0, 0.0)


def test_fire_elemental_takes_zero_fire():
    assert damage_dealt_to_mob("fire_elemental", (0.0, 4.0, 0.0)) == 0


def test_zombie_takes_full_damage():
    assert damage_dealt_to_mob("zombie", (3.0, 0.0, 0.0)) == 3


def test_knight_half_phys():
    assert damage_dealt_to_mob("knight", (4.0, 0.0, 0.0)) == 2
