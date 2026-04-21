"""Classics suite: 31 original/classic-inspired text-rendered games.

Importing this module registers all classics environments with Gymnasium.
"""

# Each game module registers its own env IDs on import.
from glyphbench.envs.classics import (  # noqa: F401
    artillery,
    battleship,
    bomberman,
    connect_four,
    farm_sim,
    flappy,
    flood_fill,
    frogger,
    gravity_maze,
    guard_evasion,
    ice_sliding,
    lights_out,
    lunar_lander,
    match3,
    maze_runner,
    memory_match,
    minesweeper,
    mirror_laser,
    nim,
    nonogram,
    pipe_connect,
    platformer,
    rush_hour,
    ski,
    slide2048,
    snake,
    sokoban,
    tetris,
    tower_defense,
    tron,
    warehouse,
    wave_defense,
)
