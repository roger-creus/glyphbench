"""Tests for MiniGrid object types."""

from __future__ import annotations


class TestWorldObject:
    def test_wall_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Wall
        w = Wall()
        assert w.char == "#"
        assert w.can_overlap is False
        assert w.can_pickup is False
        assert w.obj_type == "wall"

    def test_floor_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Floor
        f = Floor()
        assert f.char == "."
        assert f.can_overlap is True

    def test_goal_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Goal
        g = Goal()
        assert g.char == "G"
        assert g.can_overlap is True

    def test_lava_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Lava
        lv = Lava()
        assert lv.char == "L"
        assert lv.can_overlap is True

    def test_water_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Water
        wt = Water()
        assert wt.char == "~"
        assert wt.can_overlap is True

    def test_door_closed(self) -> None:
        from atlas_rl.envs.minigrid.objects import Door
        d = Door(color="red")
        assert d.char == "D"
        assert d.is_open is False
        assert d.is_locked is False
        assert d.can_overlap is False
        assert d.color == "red"

    def test_door_open(self) -> None:
        from atlas_rl.envs.minigrid.objects import Door
        d = Door(color="red")
        d.toggle(carrying=None)
        assert d.is_open is True
        assert d.char == "d"
        assert d.can_overlap is True

    def test_door_locked_needs_key(self) -> None:
        from atlas_rl.envs.minigrid.objects import Door, Key
        d = Door(color="red", is_locked=True)
        d.toggle(carrying=None)
        assert d.is_locked is True
        assert d.is_open is False
        wrong_key = Key(color="blue")
        d.toggle(carrying=wrong_key)
        assert d.is_locked is True
        right_key = Key(color="red")
        d.toggle(carrying=right_key)
        assert d.is_locked is False
        assert d.is_open is True

    def test_key_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Key
        k = Key(color="blue")
        assert k.char == "K"
        assert k.can_pickup is True
        assert k.color == "blue"

    def test_ball_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Ball
        b = Ball(color="green")
        assert b.char == "O"
        assert b.can_pickup is True
        assert b.color == "green"

    def test_box_properties(self) -> None:
        from atlas_rl.envs.minigrid.objects import Box
        b = Box(color="yellow")
        assert b.char == "B"
        assert b.can_pickup is True
        assert b.color == "yellow"

    def test_color_suffix(self) -> None:
        from atlas_rl.envs.minigrid.objects import COLOR_TO_SUFFIX, Key
        assert COLOR_TO_SUFFIX["red"] == "R"
        assert COLOR_TO_SUFFIX["green"] == "G"
        assert COLOR_TO_SUFFIX["blue"] == "B"
        k = Key(color="red")
        assert k.render_char() == "K"
        assert k.legend_name() == "key (red)"

    def test_door_render_char_variants(self) -> None:
        from atlas_rl.envs.minigrid.objects import Door
        d_closed = Door(color="red")
        assert d_closed.render_char() == "D"
        d_open = Door(color="red")
        d_open.toggle(carrying=None)
        assert d_open.render_char() == "d"
