"""Tests for MiniHack creature types."""

from glyphbench.envs.minihack.creatures import KOBOLD, Creature


class TestCreatures:
    def test_kobold_properties(self) -> None:
        assert KOBOLD.name == "kobold"
        assert KOBOLD.char == "k"
        assert KOBOLD.max_hp == 4

    def test_spawn_creature(self) -> None:
        c = Creature.spawn(KOBOLD, 3, 4)
        assert c.hp == KOBOLD.max_hp
        assert c.x == 3
        assert c.y == 4
