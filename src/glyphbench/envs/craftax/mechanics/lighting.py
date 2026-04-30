"""Phase-β lightmap subsystem.

Per-tile float lightmap in [0, 1] used by the renderer to gate visibility.
Mirrors upstream constants.py:588-590 (TORCH_LIGHT_MAP) and the per-floor
biome-light defaults from world_gen_configs.py.
"""
from __future__ import annotations

import numpy as np


TORCH_RADIUS: int = 5  # tiles


def torch_falloff(distance: float) -> float:
    """Light contribution from a single torch at this distance.
    Mirrors upstream: clip(1 - dist/5, 0, 1)."""
    return max(0.0, 1.0 - distance / TORCH_RADIUS)


def compute_lightmap(
    grid_h: int,
    grid_w: int,
    torch_positions: set[tuple[int, int]],
    biome_baseline: float,
) -> np.ndarray:
    """Compute the per-tile lightmap for one floor.

    Args:
        grid_h, grid_w: floor dimensions.
        torch_positions: set of (x, y) coords of placed torches.
        biome_baseline: per-floor base light level (0.0 unlit, 1.0 lit).

    Returns:
        np.ndarray of shape (grid_h, grid_w), values in [0, 1].
    """
    light = np.full((grid_h, grid_w), biome_baseline, dtype=np.float32)
    for (tx, ty) in torch_positions:
        # Stamp torch radius into light map.
        for dy in range(-TORCH_RADIUS, TORCH_RADIUS + 1):
            for dx in range(-TORCH_RADIUS, TORCH_RADIUS + 1):
                ny, nx = ty + dy, tx + dx
                if 0 <= ny < grid_h and 0 <= nx < grid_w:
                    # Manhattan distance per upstream falloff.
                    dist = abs(dx) + abs(dy)
                    contribution = torch_falloff(dist)
                    light[ny, nx] = min(1.0, light[ny, nx] + contribution)
    return light


VISIBILITY_THRESHOLD: float = 0.05
