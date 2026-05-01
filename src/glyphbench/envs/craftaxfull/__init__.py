"""craftaxfull suite — open-ended Crafter and Craftax full-game envs."""
from __future__ import annotations

from glyphbench.core.registry import register_env
from glyphbench.envs.craftaxfull.classic import CraftaxClassicEnv
from glyphbench.envs.craftaxfull.full import CraftaxFullEnv

REGISTRY = {
    "glyphbench/craftaxfull-classic-v0": CraftaxClassicEnv,
    "glyphbench/craftaxfull-v0": CraftaxFullEnv,
}

for env_id, cls in REGISTRY.items():
    register_env(env_id, cls)
