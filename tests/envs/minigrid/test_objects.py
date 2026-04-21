"""Tests for MiniGrid object types."""

from __future__ import annotations


class TestWorldObject:
    def test_wall_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Wall
        w = Wall()
        assert w.char == "█"
        assert w.can_overlap is False
        assert w.can_pickup is False
        assert w.obj_type == "wall"

    def test_floor_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Floor
        f = Floor()
        assert f.char == "."
        assert f.can_overlap is True

    def test_goal_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Goal
        g = Goal()
        assert g.char == "★"
        assert g.can_overlap is True

    def test_lava_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Lava
        lv = Lava()
        assert lv.char == "♧"
        assert lv.can_overlap is True

    def test_water_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Water
        wt = Water()
        assert wt.char == "≈"
        assert wt.can_overlap is True

    def test_door_closed(self) -> None:
        from glyphbench.envs.minigrid.objects import Door
        d = Door(color="red")
        assert d.char == "╣"
        assert d.is_open is False
        assert d.is_locked is False
        assert d.can_overlap is False
        assert d.color == "red"

    def test_door_open(self) -> None:
        from glyphbench.envs.minigrid.objects import Door
        d = Door(color="red")
        d.toggle(carrying=None)
        assert d.is_open is True
        assert d.char == "┤"
        assert d.can_overlap is True

    def test_door_locked_needs_key(self) -> None:
        from glyphbench.envs.minigrid.objects import Door, Key
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
        from glyphbench.envs.minigrid.objects import Key
        k = Key(color="blue")
        assert k.can_pickup is True
        assert k.color == "blue"
        assert len(k.render_char()) == 1  # single char per color

    def test_ball_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Ball
        b = Ball(color="green")
        assert b.can_pickup is True
        assert b.color == "green"
        assert len(b.render_char()) == 1

    def test_box_properties(self) -> None:
        from glyphbench.envs.minigrid.objects import Box
        b = Box(color="yellow")
        assert b.can_pickup is True
        assert b.color == "yellow"
        assert len(b.render_char()) == 1

    def test_color_suffix(self) -> None:
        from glyphbench.envs.minigrid.objects import Key
        k_red = Key(color="red")
        assert k_red.render_char() == "♠"
        assert k_red.legend_name() == "key (red)"
        # Different colors produce distinct chars
        k_green = Key(color="green")
        assert k_green.render_char() == "♤"
        assert k_green.render_char() != k_red.render_char()

    def test_door_render_char_variants(self) -> None:
        from glyphbench.envs.minigrid.objects import Door
        d_closed = Door(color="red")
        assert d_closed.render_char() == "╣"
        d_open = Door(color="red")
        d_open.toggle(carrying=None)
        assert d_open.render_char() == "┤"
        # Different colors produce distinct chars
        d_yellow = Door(color="yellow")
        assert d_yellow.render_char() != d_closed.render_char()
