"""atlas_rl — unified ASCII RL benchmark for LLM evaluation.

Importing this package triggers env registration for every suite subpackage.
"""

__version__ = "0.0.1"

# Importing each envs subpackage triggers gym registration via that package's __init__.
from atlas_rl.envs import dummy as _dummy_envs  # noqa: F401
from atlas_rl.envs import minigrid as _minigrid_envs  # noqa: F401
from atlas_rl.envs import minihack as _minihack_envs  # noqa: F401
from atlas_rl.envs import atari as _atari_envs  # noqa: F401
from atlas_rl.envs import procgen as _procgen_envs  # noqa: F401
from atlas_rl.envs import craftax as _craftax_envs  # noqa: F401
