"""ActionSpec: per-env action vocabulary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Each env owns its own Discrete(N) action space with per-game action names.

    The harness does not filter actions by turn-validity; the LLM must reason
    about which actions make sense in the current state. Action names are
    SHOUTY_SNAKE_CASE by convention so the parser can extract them with a
    simple regex as a fallback to JSON parsing.
    """

    names: tuple[str, ...]
    descriptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.descriptions):
            raise ValueError(
                f"ActionSpec.names and ActionSpec.descriptions must have the same length, "
                f"got {len(self.names)} and {len(self.descriptions)}"
            )
        if len(set(self.names)) != len(self.names):
            dupes = [n for n in self.names if self.names.count(n) > 1]
            raise ValueError(f"ActionSpec.names must be unique; duplicates: {dupes}")

    @property
    def n(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        """Look up an action by name.

        Tries an exact match first, then a whitespace-stripped case-insensitive
        fallback (only accepted when the case-folded match is unique — a
        collision with another action under case-folding raises KeyError).
        """
        stripped = name.strip()
        try:
            return self.names.index(stripped)
        except ValueError:
            pass

        folded = stripped.casefold()
        matches = [i for i, n in enumerate(self.names) if n.casefold() == folded]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(
            f"Unknown action name: {name!r}. Valid: {list(self.names)}"
        )

    def render_for_prompt(self) -> str:
        """Rendered once into the system prompt at episode start, not per turn."""
        lines = ["Actions (pick exactly one per turn):"]
        for name, desc in zip(self.names, self.descriptions, strict=True):
            lines.append(f"  {name} — {desc}")
        return "\n".join(lines)
