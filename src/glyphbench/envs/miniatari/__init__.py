"""miniatari suite — short-horizon, [-1, 1]-bounded redesigns of atari games.

Each env follows the miniaturization recipe in
docs/superpowers/specs/2026-05-01-miniatari-and-reward-normalization-design.md
§1: smaller play field, tight terminal win condition, max_turns in
[100, 500], structural [-1, 1] reward shape.
"""
from __future__ import annotations

from glyphbench.core.registry import register_env

from glyphbench.envs.miniatari.pong import MiniPongEnv

REGISTRY: dict = {
    "glyphbench/miniatari-pong-v0": MiniPongEnv,
}

for env_id, cls in REGISTRY.items():
    register_env(env_id, cls)
