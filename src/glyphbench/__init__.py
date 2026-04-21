"""glyphbench — unified ASCII RL benchmark for LLM evaluation.

Importing this package triggers env registration for every suite subpackage.
"""

__version__ = "0.1.0"

# Importing each envs subpackage triggers gym registration via that package's __init__.
from glyphbench.envs import atari as _atari_envs  # noqa: F401
from glyphbench.envs import craftax as _craftax_envs  # noqa: F401
from glyphbench.envs import dummy as _dummy_envs  # noqa: F401
from glyphbench.envs import minigrid as _minigrid_envs  # noqa: F401
from glyphbench.envs import minihack as _minihack_envs  # noqa: F401
from glyphbench.envs import procgen as _procgen_envs  # noqa: F401
from glyphbench.envs import classics as _classics_envs  # noqa: F401
