"""Environment implementations. Importing a subpackage registers its envs."""

from __future__ import annotations

ALL_SUITES: tuple[str, ...] = (
    "dummy",
    "minigrid",
    "minihack",
    "atari",
    "craftax",
    "procgen",
    "classics",
)


def _import_all_suites() -> None:
    """Force-import every suite so side-effect registration populates REGISTRY.

    Suites that aren't yet ported (or aren't installed in this build) are
    skipped silently — ImportError covers missing modules, TypeError tolerates
    any suite still on the legacy gym-style register_env signature during the
    migration window.
    """
    for suite in ALL_SUITES:
        try:
            __import__(f"glyphbench.envs.{suite}")
        except (ImportError, TypeError):
            # Suite not present in this build — acceptable during migration.
            pass
