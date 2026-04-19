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


# Distinct single-character render for each (type, color) combination.
# This lets the LLM distinguish same-type objects of different colors.
_KEY_CHARS: dict[str, str] = {
    "red": "K", "green": "k", "blue": "j", "yellow": "Y", "purple": "y", "grey": "J",
}
_BALL_CHARS: dict[str, str] = {
    "red": "O", "green": "o", "blue": "Q", "yellow": "q", "purple": "0", "grey": "9",
}
_BOX_CHARS: dict[str, str] = {
    "red": "B", "green": "b", "blue": "P", "yellow": "p", "purple": "8", "grey": "7",
}
_DOOR_CLOSED_CHARS: dict[str, str] = {
    "red": "D", "green": "A", "blue": "E", "yellow": "F", "purple": "H", "grey": "I",
}
_DOOR_OPEN_CHARS: dict[str, str] = {
    "red": "d", "green": "a", "blue": "e", "yellow": "f", "purple": "h", "grey": "i",
}


@dataclass
class Key(WorldObject):
    color: str = "red"

    def __post_init__(self) -> None:
        self.obj_type = "key"
        self.char = _KEY_CHARS.get(self.color, "K")
        self.can_overlap = False
        self.can_pickup = True

    def render_char(self) -> str:
        return _KEY_CHARS.get(self.color, "K")


@dataclass
class Ball(WorldObject):
    color: str = "red"

    def __post_init__(self) -> None:
        self.obj_type = "ball"
        self.char = _BALL_CHARS.get(self.color, "O")
        self.can_overlap = False
        self.can_pickup = True

    def render_char(self) -> str:
        return _BALL_CHARS.get(self.color, "O")


@dataclass
class Box(WorldObject):
    color: str = "red"
    contains: WorldObject | None = None

    def __post_init__(self) -> None:
        self.obj_type = "box"
        self.char = _BOX_CHARS.get(self.color, "B")
        self.can_overlap = False
        self.can_pickup = True

    def render_char(self) -> str:
        return _BOX_CHARS.get(self.color, "B")


@dataclass
class Door(WorldObject):
    color: str = "red"
    is_open: bool = False
    is_locked: bool = False

    def __post_init__(self) -> None:
        self.obj_type = "door"
        self.char = (
            _DOOR_OPEN_CHARS.get(self.color, "d") if self.is_open
            else _DOOR_CLOSED_CHARS.get(self.color, "D")
        )
        self.can_overlap = self.is_open
        self.can_pickup = False

    def toggle(self, carrying: WorldObject | None) -> bool:
        """Toggle the door. Returns True if state changed."""
        if self.is_locked:
            if isinstance(carrying, Key) and carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                self.char = _DOOR_OPEN_CHARS.get(self.color, "d")
                self.can_overlap = True
                return True
            return False
        self.is_open = not self.is_open
        self.char = (
            _DOOR_OPEN_CHARS.get(self.color, "d") if self.is_open
            else _DOOR_CLOSED_CHARS.get(self.color, "D")
        )
        self.can_overlap = self.is_open
        return True

    def render_char(self) -> str:
        return self.char
