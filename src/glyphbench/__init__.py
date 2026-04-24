"""glyphbench — unified benchmark of text-rendered RL environments.

Importing this package triggers env registration for every suite subpackage
that has already been ported to the new class-object registry. Suites still
on the old API are silently skipped during the migration.
"""

__version__ = "0.1.0"

# Suites populate REGISTRY on import. During the migration, safely skip any
# suite that still uses the old gym-style register_env signature.
for _suite in ("dummy", "minigrid", "minihack", "atari", "craftax", "procgen", "classics"):
    try:
        __import__(f"glyphbench.envs.{_suite}")
    except (ImportError, TypeError):
        pass
