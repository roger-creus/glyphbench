"""GlyphBench: unified benchmark of 292 text-rendered RL environments."""

__version__ = "0.1.0"

from glyphbench.core import (
    ActionSpec,
    BaseGlyphEnv,
    GridObservation,
    REGISTRY,
    all_glyphbench_env_ids,
    make_env,
    register_env,
)
from glyphbench.verifiers_integration import (
    GlyphbenchMultiTurnEnv,
    GlyphbenchXMLParser,
    EpisodicReturnRubric,
    load_environment,
)

# Importing any suite module populates REGISTRY eagerly:
from glyphbench.envs import dummy  # noqa: F401

# The rest of the suites are optional during migration; they get added as
# the ports land in M2-M4 (import safely, silently skip on ImportError or
# legacy-register_env TypeErrors).
# TEMPORARY — remove at end of M4 (plan Task 4.7)
for _suite in ("minigrid", "minihack", "atari", "craftax", "procgen", "classics"):
    try:
        __import__(f"glyphbench.envs.{_suite}")
    except (ImportError, TypeError):
        pass

__all__ = [
    "ActionSpec",
    "BaseGlyphEnv",
    "GridObservation",
    "GlyphbenchMultiTurnEnv",
    "GlyphbenchXMLParser",
    "EpisodicReturnRubric",
    "REGISTRY",
    "all_glyphbench_env_ids",
    "make_env",
    "register_env",
    "load_environment",
]
