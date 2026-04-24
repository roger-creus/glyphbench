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

# Importing any suite module populates REGISTRY via register_env side-effects.
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
