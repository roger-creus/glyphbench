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

# Importing any suite module populates REGISTRY eagerly. Suites still on the
# legacy register_env signature (or not yet ported) are skipped silently —
# acceptable during the migration window.
# TEMPORARY — remove at end of M4 (plan Task 4.7)
from glyphbench.envs import _import_all_suites as _load_suites

_load_suites()
del _load_suites

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
