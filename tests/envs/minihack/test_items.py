"""Tests for MiniHack items."""

from __future__ import annotations

from glyphbench.envs.minihack.items import (
    APPLE,
    CORPSE,
    DAGGER,
    FOOD_RATION,
    POTION_HEALING,
    POTION_LEVITATION,
    POTION_SPEED,
    RING_LEVITATION,
    SCROLL_IDENTIFY,
    SCROLL_LIGHT,
    SCROLL_TELEPORT,
    SWORD,
    WAND_COLD,
    WAND_DEATH,
    WAND_FIRE,
)


class TestItems:
    def test_food_properties(self) -> None:
        assert FOOD_RATION.char == "%"
        assert FOOD_RATION.item_type == "food"
        assert FOOD_RATION.legend_name() == "food ration"

    def test_all_food_items(self) -> None:
        for item in (FOOD_RATION, APPLE, CORPSE):
            assert item.char == "%"
            assert item.item_type == "food"

    def test_weapon_properties(self) -> None:
        assert SWORD.char == ")"
        assert SWORD.item_type == "weapon"
        assert DAGGER.char == ")"
        assert DAGGER.item_type == "weapon"

    def test_wand_properties(self) -> None:
        for item in (WAND_DEATH, WAND_FIRE, WAND_COLD):
            assert item.char == "/"
            assert item.item_type == "wand"

    def test_scroll_properties(self) -> None:
        for item in (SCROLL_IDENTIFY, SCROLL_TELEPORT, SCROLL_LIGHT):
            assert item.char == "?"
            assert item.item_type == "scroll"

    def test_potion_properties(self) -> None:
        for item in (POTION_HEALING, POTION_SPEED, POTION_LEVITATION):
            assert item.char == "!"
            assert item.item_type == "potion"

    def test_ring_properties(self) -> None:
        assert RING_LEVITATION.char == "="
        assert RING_LEVITATION.item_type == "ring"

    def test_legend_name(self) -> None:
        assert SWORD.legend_name() == "long sword"
        assert WAND_DEATH.legend_name() == "wand of death"
