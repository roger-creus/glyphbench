# GlyphBench

A unified benchmark of **210 text-rendered RL environments** for evaluating LLM agents on sequential decision-making.

Every environment renders its state as a Unicode text grid with a legend, HUD, and discrete named actions. Observations are deterministic (seeded), making results fully reproducible.

## Installation

```bash
# Core (environments only)
uv add glyphbench

# With eval infrastructure (vLLM offline batched inference)
uv add "glyphbench[eval]"

# Everything (providers + analysis + dev)
uv add "glyphbench[all]"
```

From source:

```bash
git clone https://github.com/roger-creus/glyphbench.git
cd glyphbench
uv sync --all-extras
```

## Quick Start

```python
import glyphbench  # registers all 210 environments
import gymnasium as gym

env = gym.make("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(seed=42)
print(obs)
```

```
[Legend]
· — floor
→ — you, facing right
█ — wall
★ — goal

[Grid]
███████
█→····█
█·····█
█·····█
█····★█
███████
```

## Environment Suites

| Suite | Envs | Description | Actions |
|-------|------|-------------|---------|
| **MiniGrid** | 71 | Grid navigation, key/door puzzles, dynamic obstacles, memory | 7 |
| **MiniHack** | 63 | NetHack-inspired dungeon crawling, combat, items, skills | 22 |
| **Atari** | 57 | Classic arcade games (Pong, Breakout, Space Invaders, ...) | 3-18 |
| **Procgen** | 16 | Procedurally generated platformers, shooters, mazes | 4-6 |
| **Craftax** | 2 | Open-world survival crafting with 22+ achievements | 19 |

All environments use Unicode glyphs (`→↓←↑` for player direction, `█` walls, `★` goals, `≈` water, etc.) to minimize symbol collisions and maximize readability for both humans and LLMs.

## Observation Format

Every environment returns a text observation with four sections:

- **[Legend]** -- maps each glyph to its meaning (e.g., `★ — goal`)
- **[HUD]** -- game state (HP, score, inventory, step count)
- **[Grid]** -- the 2D Unicode grid
- **[Message]** -- optional per-turn narrative

Each environment also provides `system_prompt()` with full game rules, action descriptions, and reward structure -- designed to be passed directly to an LLM as a system message.

## Running LLM Evaluations

GlyphBench includes a batched evaluation runner using vLLM offline inference:

```bash
# Evaluate Qwen3.5-4B on all 210 environments (25 episodes each)
uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-4B \
    --episodes 25 \
    --output results/

# Single suite
uv run python eval/run_eval.py --model Qwen/Qwen3.5-4B --suites atari

# Specific envs
uv run python eval/run_eval.py --model Qwen/Qwen3.5-4B \
    --envs glyphbench/minigrid-doorkey-6x6-v0 glyphbench/atari-pong-v0
```

Results include per-env metrics (mean return, episode length, parse failure rate) plus full trajectory recordings (observation, LLM response, action, reward at every step).

See `eval/README.md` for cluster deployment with SLURM.

## Trajectory Replay

```bash
# Replay a trajectory with Unicode + color rendering
uv run python scripts/replay_trajectory.py results/trajectories/glyphbench__atari-pong-v0/seed_42_ep_0.jsonl

# Export as GIF
uv run python scripts/replay_trajectory.py trajectory.jsonl --gif output.gif
```

## Interactive Demo

```bash
# Watch random agent play all envs (5 steps each, pause between)
uv run python scripts/demo_all_envs.py --steps 5 --pause

# Single env, slower playback
uv run python scripts/demo_all_envs.py --env glyphbench/craftax-classic-v0 --delay 0.2
```

## Using GlyphBench with Your Own Agent

```python
import glyphbench
import gymnasium as gym

env = gym.make("glyphbench/minigrid-doorkey-6x6-v0")
obs, info = env.reset(seed=42)

done = False
total_reward = 0.0

while not done:
    # Your agent: parse obs, pick action by name
    action = your_agent(obs, env.unwrapped.action_spec.names)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode return: {total_reward}")
```

## Project Structure

```
src/glyphbench/
    core/       # BaseAsciiEnv, GridObservation, ActionSpec, registry
    envs/       # 5 suites, 210 environments
    harness/    # LLM agent loop, prompt builder, JSON parser
    providers/  # vLLM, OpenAI, Anthropic, Gemini clients
    runner/     # Async benchmark runner, config, dashboard, storage
    plotting/   # Analysis and visualization
eval/           # Batched vLLM eval runner + cluster manager
scripts/        # Demo viewer, trajectory replay, GIF export
examples/       # Quickstart, random agent, custom agent loop
```

## Development

```bash
uv sync --all-extras
uv run pytest               # 2255 tests
uv run ruff check src/
uv run mypy src/glyphbench/
```

## Citation

```bibtex
@article{glyphbench2026,
  title={GlyphBench: A Unified Benchmark for Evaluating LLM Agents on Sequential Decision-Making},
  author={Creus Castanyer, Roger},
  year={2026},
}
```

## License

MIT
