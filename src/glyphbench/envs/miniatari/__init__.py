"""miniatari suite — short-horizon, [-1, 1]-bounded redesigns of atari games.

Each env follows the miniaturization recipe in
docs/superpowers/specs/2026-05-01-miniatari-and-reward-normalization-design.md
§1: smaller play field, tight terminal win condition, max_turns in
[100, 500], structural [-1, 1] reward shape.
"""
from __future__ import annotations

from glyphbench.core.registry import register_env

from glyphbench.envs.miniatari.boxing import MiniBoxingEnv
from glyphbench.envs.miniatari.breakout import MiniBreakoutEnv
from glyphbench.envs.miniatari.doubledunk import MiniDoubleDunkEnv
from glyphbench.envs.miniatari.fishingderby import MiniFishingDerbyEnv
from glyphbench.envs.miniatari.icehockey import MiniIceHockeyEnv
from glyphbench.envs.miniatari.pong import MiniPongEnv
from glyphbench.envs.miniatari.surround import MiniSurroundEnv
from glyphbench.envs.miniatari.tennis import MiniTennisEnv

REGISTRY: dict = {
    "glyphbench/miniatari-pong-v0": MiniPongEnv,
    "glyphbench/miniatari-tennis-v0": MiniTennisEnv,
    "glyphbench/miniatari-icehockey-v0": MiniIceHockeyEnv,
    "glyphbench/miniatari-doubledunk-v0": MiniDoubleDunkEnv,
    "glyphbench/miniatari-fishingderby-v0": MiniFishingDerbyEnv,
    "glyphbench/miniatari-boxing-v0": MiniBoxingEnv,
    "glyphbench/miniatari-surround-v0": MiniSurroundEnv,
    "glyphbench/miniatari-breakout-v0": MiniBreakoutEnv,
}

for env_id, cls in REGISTRY.items():
    register_env(env_id, cls)
