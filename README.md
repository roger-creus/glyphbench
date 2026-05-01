<div align="center">

# ★  →  GlyphBench  ←  ★

**A benchmark of 343 text-rendered reinforcement-learning environments for evaluating LLM agents on sequential decision-making.**

<img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/hero.gif" width="320" />

[Leaderboard](https://roger-creus.github.io/glyphbench/leaderboard/) · [Paper (coming soon)](#) · [Quickstart](#quickstart) · [Contributing](CONTRIBUTING.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-green)](https://roger-creus.github.io/glyphbench/leaderboard/)

</div>

Every environment renders its state as a Unicode text grid with a legend and discrete named actions. The agent sees only the grid — no privileged state-channel — so every game-relevant fact must be readable off the glyphs themselves. Observations are deterministic (seeded), making results fully reproducible.

## At a glance

| Suite | Envs | What it tests | Actions |
|---|---:|---|---|
| MiniGrid | 71 | Grid navigation, key/door puzzles, dynamic obstacles, memory | 7 |
| MiniHack | 63 | NetHack-inspired dungeons, combat, items, skills | 22 |
| Atari | 57 | Classic arcade (Pong, Breakout, Space Invaders, …) — long-horizon archival | 3-10 |
| Classics | 50 | Snake, Sokoban, Minesweeper, Sudoku, Nim, … | 2-256 |
| Miniatari | 43 | Short-horizon redesigns of arcade Atari (born `[-1, 1]`-compliant) | 3-10 |
| Craftax | 41 | Open-world survival + crafting, dungeon floors, focused sub-tasks | 19 |
| Procgen | 16 | Procedurally generated platformers, shooters, mazes | 4-6 |
| Craftaxfull | 2 | Open-ended Crafter / Craftax (long-horizon archival) | 19 / 45 |

All environments use single-codepoint Unicode glyphs (`→↓←↑` for player direction, `█` walls, `★` goals, `≈` water, …) with no symbol collisions inside a suite. Every env's cumulative episodic return is structurally bounded to **`[-1, 1]`** so per-task results are directly comparable.

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

| **FirstDay** | **FightCow** | **ChopTrees** | **Speedrun** | **FindWater** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-firstday-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-fight-cow-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-choptrees-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-speedrun-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-find-water-v0.gif" width="130" /> |

### Procgen

| **CoinRun** | **Maze** | **Bigfish** | **Ninja** | **Jumper** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-coinrun-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-maze-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-bigfish-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-ninja-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-jumper-v0.gif" width="130" /> |

### Miniatari

| **Pong** | **Breakout** | **SpaceInvaders** | **Freeway** | **MsPacman** |
|:---:|:---:|:---:|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__miniatari-pong-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__miniatari-breakout-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__miniatari-spaceinvaders-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__miniatari-freeway-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__miniatari-mspacman-v0.gif" width="130" /> |

Atari originals run at `max_turns=10000` and were designed for thousands of frames per game — prohibitive for LLM evaluation, where each step is an API call. **Miniatari is a curated suite of 43 short-horizon redesigns** that keeps each game's identity (mechanics, theme, core decision) while compressing the horizon to `max_turns ∈ [100, 500]`. It's the practical default for LLM eval and training; the original `atari` suite is kept as a long-horizon archival reference and excluded from default eval.

The miniaturization recipe:

- **Smaller play field** — typically 12×8 to 16×16 (vs ~20×20 in the originals).
- **Tight terminal win condition** — clear 6 bricks (vs full 60-brick wall), first-to-3 (vs first-to-21), destroy 5 asteroids (vs play-forever-and-score).
- **No frame-skip** — 1 LLM action = 1 tick. Compression comes from smaller grids and tighter wins, not time dilation.
- **Structurally bounded reward `[-1, 1]`** — every per-step reward is `+1/N` per progress unit (Pattern A), `±1/W` per agent/opponent point (Pattern C, adversarial), or progress with terminal `−1` on death (Pattern D). The cumulative episodic return is `[-1, 1]` by construction.
- **Calibrated against random rollouts** — each env's random success rate / mean length / mean return is recorded in its docstring; no env is trivially solvable.

14 atari games are dropped from miniatari because their identity is *exploration over an unbounded map* (`montezumarevenge`, `pitfall`, `privateeye`, `solaris`, `gravitar`, `venture`, `hero`), *endurance over a long horizon* (`crazyclimber`, `videopinball`, `roadrunner`), *multi-screen multi-stage* (`krull`, `jamesbond`), or *no clear terminal win* (`tutankham`, `namethisgame`). Those stay in `atari/` for archival eval.

### Craftaxfull

| **Crafter (Classic)** | **Craftax (Full)** |
|:---:|:---:|
| <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftaxfull-classic-v0.gif" width="130" /> | <img src="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftaxfull-v0.gif" width="130" /> |

The two upstream-faithful Crafter / Craftax games — `craftaxfull-classic-v0` (22 achievements) and `craftaxfull-v0` (93 achievements) — extracted from the `craftax` suite into their own home. They use `max_turns=10000` and the full open-ended reward shape (per-achievement milestone summing to 1.0, terminal `−1` on death). Too long for default LLM eval, but kept as the open-ended capability ceiling.

For day-to-day evaluation and training, the **41 `craftax/` subtask envs** (firstday, iron-bootstrap, fight-zombie, build-shelter, craftpickaxe, …) cover the same mechanics in a 100-300 turn budget per task. Default `eval/run_full.sh` excludes `craftaxfull`; opt in with `eval/run_archival.sh`.

→ [Browse all 343 environments on the leaderboard gallery](https://roger-creus.github.io/glyphbench/leaderboard/)

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

### One env per suite

```python
from glyphbench.core import make_env

# MiniGrid — grid navigation, key/door puzzles
make_env("glyphbench/minigrid-doorkey-6x6-v0")

# MiniHack — NetHack-inspired dungeons
make_env("glyphbench/minihack-corridor-r3-v0")

# Atari — full-horizon arcade originals (archival)
make_env("glyphbench/atari-pong-v0")

# Miniatari — short-horizon arcade redesigns (recommended for eval/training)
make_env("glyphbench/miniatari-pong-v0")

# Classics — Snake, Sokoban, 2048, Tetris, Minesweeper, ...
make_env("glyphbench/classics-snake-medium-v0")

# Craftax — survival + crafting subtasks
make_env("glyphbench/craftax-craftpickaxe-v0")

# Craftaxfull — open-ended Crafter / Craftax (long-horizon archival)
make_env("glyphbench/craftaxfull-classic-v0")

# Procgen — procedurally generated platformers, shooters, mazes
make_env("glyphbench/procgen-coinrun-v0")
```

### Filtering by suite or task pattern

`load_environment` accepts include/exclude filters that select a subset of the registry — useful for training configs and eval scripts.

```python
import glyphbench

# Train on miniatari + minigrid + minihack only:
env = glyphbench.load_environment(
    include_suites=["miniatari", "minigrid", "minihack"],
    num_episodes=5,
)

# Eval everything except the long-horizon archival suites (this is the default in eval/run_full.sh):
env = glyphbench.load_environment(
    exclude_suites=["atari", "craftaxfull"],
    num_episodes=5,
)

# Eval all atari pong-style games via fnmatch pattern:
env = glyphbench.load_environment(
    include_tasks=["glyphbench/*-pong-v0"],
    num_episodes=5,
)
```

CLI counterparts in `eval/run_full.sh` (default-filtered) and `eval/run_archival.sh` (atari + craftaxfull only).

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
