"""MiniGrid world objects: walls, doors, keys, balls, boxes, lava, goals, water.

Each object type defines its ASCII render char, color, and interaction rules.
Objects are mutable (doors can open/close) but position is tracked by the grid,
not by the object itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COLOR_TO_SUFFIX: dict[str, str] = {
    "red": "R",
    "green": "G",
    "blue": "B",
    "yellow": "Y",
    "purple": "P",
    "grey": "W",
}

COLORS: tuple[str, ...] = tuple(COLOR_TO_SUFFIX.keys())


@dataclass
class WorldObject:
    """Base for all grid objects."""

    obj_type: str = field(init=False)
    char: str = field(init=False)
    color: str = ""
    can_overlap: bool = field(init=False, default=False)
    can_pickup: bool = field(init=False, default=False)

    def render_char(self) -> str:
        return self.char

    def legend_name(self) -> str:
        if self.color:
            return f"{self.obj_type} ({self.color})"
        return self.obj_type


@dataclass
class Wall(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "wall"
        self.char = "#"
        self.can_overlap = False
        self.can_pickup = False


@dataclass
class Floor(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "floor"
        self.char = "."
        self.can_overlap = True
        self.can_pickup = False


@dataclass
class Goal(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "goal"
        self.char = "G"
        self.can_overlap = True
        self.can_pickup = False


@dataclass
class Lava(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "lava"
        self.char = "L"
        self.can_overlap = True
        self.can_pickup = False


@dataclass
class Water(WorldObject):
    def __post_init__(self) -> None:
        self.obj_type = "water"
        self.char = "~"
        self.can_overlap = True
        self.can_pickup = False


@dataclass
class Key(WorldObject):
    color: str = "red"

    def __post_init__(self) -> None:
        self.obj_type = "key"
        self.char = "K"
        self.can_overlap = False
        self.can_pickup = True


@dataclass
class Ball(WorldObject):
    color: str = "red"

    def __post_init__(self) -> None:
        self.obj_type = "ball"
        self.char = "O"
        self.can_overlap = False
        self.can_pickup = True


@dataclass
class Box(WorldObject):
    color: str = "red"
    contains: WorldObject | None = None

    def __post_init__(self) -> None:
        self.obj_type = "box"
        self.char = "B"
        self.can_overlap = False
        self.can_pickup = True


@dataclass
class Door(WorldObject):
    color: str = "red"
    is_open: bool = False
    is_locked: bool = False

    def __post_init__(self) -> None:
        self.obj_type = "door"
        self.char = "d" if self.is_open else "D"
        self.can_overlap = self.is_open
        self.can_pickup = False

    def toggle(self, carrying: WorldObject | None) -> bool:
        """Toggle the door. Returns True if state changed."""
        if self.is_locked:
            if isinstance(carrying, Key) and carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                self.char = "d"
                self.can_overlap = True
                return True
            return False
        self.is_open = not self.is_open
        self.char = "d" if self.is_open else "D"
        self.can_overlap = self.is_open
        return True

    def render_char(self) -> str:
        return self.char
