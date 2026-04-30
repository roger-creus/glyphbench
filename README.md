<div align="center">

# ★  →  GlyphBench  ←  ★

**A benchmark of 300 text-rendered reinforcement-learning environments for evaluating LLM agents on sequential decision-making.**

<img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/hero.gif" width="320" />

[Leaderboard](https://roger-creus.github.io/glyphbench/leaderboard/) · [Paper (coming soon)](#) · [Quickstart](#quickstart) · [Contributing](CONTRIBUTING.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![vLLM compatible](https://img.shields.io/badge/inference-vLLM-orange)
![Verifiers](https://img.shields.io/badge/eval-verifiers-purple)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-green)](https://roger-creus.github.io/glyphbench/leaderboard/)

</div>

Every environment renders its state as a Unicode text grid with a legend and discrete named actions. The agent sees only the grid — no privileged state-channel — so every game-relevant fact must be readable off the glyphs themselves. Observations are deterministic (seeded), making results fully reproducible.

## At a glance

| Suite | Envs | What it tests | Actions |
|---|---:|---|---|
| MiniGrid | 71 | Grid navigation, key/door puzzles, dynamic obstacles, memory | 7 |
| MiniHack | 63 | NetHack-inspired dungeons, combat, items, skills | 22 |
| Atari | 57 | Classic arcade (Pong, Breakout, Space Invaders, …) | 3-10 |
| Classics | 50 | Snake, Sokoban, Minesweeper, Sudoku, Nim, … | 2-256 |
| Craftax | 43 | Open-world survival + crafting, dungeon floors, focused sub-tasks | 19 / 45 |
| Procgen | 16 | Procedurally generated platformers, shooters, mazes | 4-6 |

All environments use single-codepoint Unicode glyphs (`→↓←↑` for player direction, `█` walls, `★` goals, `≈` water, …) with no symbol collisions inside a suite.

## Browse the suites

### MiniGrid

| **DoorKey-6x6** | **MultiRoom-N4** | **LavaGap-S6** | **Dyn-Obstacles** | **KeyCorridor** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-doorkey-6x6-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-multiroom-n4-s5-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-lavagap-s6-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-dynamic-obstacles-16x16-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-keycorridor-s3r3-v0.gif" width="130" /> |

### MiniHack

| **Room-Monster** | **Corridor-R3** | **LavaCross** | **Eat** | **HideNSeek** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-room-monster-15x15-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-corridor-r3-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-lavacross-full-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-eat-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-hidenseek-v0.gif" width="130" /> |

### Atari

| **Pong** | **Breakout** | **SpaceInvaders** | **MsPacman** | **Frostbite** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-pong-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-breakout-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-spaceinvaders-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-mspacman-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-frostbite-v0.gif" width="130" /> |

### Classics

| **Snake** | **Sokoban** | **Minesweeper** | **Tetris** | **Flappy** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-snake-medium-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-sokoban-easy-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-minesweeper-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-tetris-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-flappy-v0.gif" width="130" /> |

### Craftax

| **Classic** | **FightCow** | **ChopTrees** | **Speedrun** | **FindWater** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-classic-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-fight-cow-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-choptrees-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-speedrun-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-find-water-v0.gif" width="130" /> |

### Procgen

| **CoinRun** | **Maze** | **Bigfish** | **Ninja** | **Jumper** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-coinrun-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-maze-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-bigfish-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-ninja-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-jumper-v0.gif" width="130" /> |

→ [Browse all 300 environments on the leaderboard gallery](https://roger-creus.github.io/glyphbench/leaderboard/)

## Install

```bash
uv add glyphbench                    # core (environments only)
uv add "glyphbench[eval]"            # + verifiers + vLLM (eval + RL integration)
uv add "glyphbench[all]"             # + providers, analysis, dev tooling
```

## Quickstart

```python
import glyphbench
from glyphbench.core import make_env

# Direct game loop
env = make_env("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(42)
print(obs)

# Or: load as a verifiers environment for eval / RL
vf_env = glyphbench.load_environment(task_id="glyphbench/minigrid-empty-5x5-v0")
```

## Trajectory replay

<img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/gb_replay.gif" width="700" />

`glyphbench replay` is a rich TUI for stepping through saved rollouts: per-turn grid + reasoning + memory + HUD. In `--pause` mode the hotkeys drop into a pager for any panel:

| Key | Opens |
|---|---|
| `s` | full system prompt |
| `r` | full reasoning chain for this turn |
| `m` | previous + updated memory (side-by-side) |
| `l` | full legend (glyph table) |
| `←` / `→` | step backward / forward |
| `q` | exit current rollout, advance to next match |

Full reference: [docs/REPLAY.md](docs/REPLAY.md).

## Documentation

- [Observation format · harness](docs/OBSERVATION_FORMAT.md)
- [LLM evaluation (vLLM / verifiers / `prime eval run`)](eval/README.md)
- [RL framework hooks (prime-rl)](src/glyphbench/rl/README.md)
- [Trajectory replay tool](docs/REPLAY.md)
- [Top-level scripts (demo, replay, GIF export, upload)](scripts/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Use with your own agent](docs/INTEGRATION.md)
- [Contributing](CONTRIBUTING.md)

## Citation

```bibtex
@article{glyphbench2026,
  title   = {GlyphBench: A Unified Benchmark for Evaluating LLM Agents on Sequential Decision-Making},
  author  = {Roger Creus Castanyer},
  year    = {2026},
}
```

## License

MIT
