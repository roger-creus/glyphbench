# GlyphBench

A benchmark of **292 text-rendered reinforcement-learning environments** for evaluating LLM agents on sequential decision-making.

Every environment renders its state as a Unicode text grid with a legend, HUD, and discrete named actions. Observations are deterministic (seeded), making results fully reproducible.

- **Leaderboard & rollouts:** [roger-creus.github.io/glyphbench](https://roger-creus.github.io/glyphbench/leaderboard/)
- **Assets (GIFs, trajectories):** [huggingface.co/datasets/anon-paper-submission/glyphbench-assets](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets)
- **Paper:** coming soon
- **Contributing a run:** see [CONTRIBUTING.md](CONTRIBUTING.md)

## At a glance

| Suite | Envs | What it tests | Actions |
|---|---:|---|---:|
| MiniGrid | 71 | Grid navigation, key/door puzzles, dynamic obstacles, memory | 7 |
| MiniHack | 63 | NetHack-inspired dungeons, combat, items, skills | 22 |
| Atari | 57 | Classic arcade (Pong, Breakout, Space Invaders, …) | 3–18 |
| Classics | 50 | Snake, Sokoban, Minesweeper, Sudoku, Nim, … | 4–10 |
| Craftax | 35 | Open-world survival + crafting, dungeon floors, focused sub-tasks | 19 |
| Procgen | 16 | Procedurally generated platformers, shooters, mazes | 4–6 |

All environments use single-codepoint Unicode glyphs (`→↓←↑` for player direction, `█` walls, `★` goals, `≈` water, …) with no symbol collisions across a suite.

## Install

```bash
uv add glyphbench                    # core (environments only)
uv add "glyphbench[eval]"            # + batched vLLM eval runner
uv add "glyphbench[all]"             # + providers, analysis, dev tooling
```

From source:

```bash
git clone https://github.com/roger-creus/glyphbench.git
cd glyphbench
uv sync --all-extras
```

## Quick start

```python
import glyphbench  # registers all 292 environments
import gymnasium as gym

env = gym.make("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(seed=42)
print(obs)
```

```
[Legend]
· floor    → you, facing right    █ wall    ★ goal

[Grid]
███████
█→····█
█·····█
█·····█
█····★█
███████
```

Every environment also exposes `env.unwrapped.system_prompt()` — a compact description of rules, actions, reward structure, and termination conditions, ready to pass as a system message to any LLM.

## Observation format

Every environment returns a single text string with four sections:

- `[Legend]` — maps each glyph to its meaning.
- `[HUD]` — game state (HP, score, inventory, step counter).
- `[Grid]` — the 2D Unicode grid.
- `[Message]` — optional per-turn narrative.

## Running LLM evaluations

GlyphBench includes a batched evaluation runner using vLLM offline inference:

```bash
# All 292 envs, 25 episodes each, history-CoT harness
uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-4B \
    --harness history_cot \
    --episodes 25 \
    --output results/

# Single suite
uv run python eval/run_eval.py --model Qwen/Qwen3.5-4B --suites atari

# Specific envs
uv run python eval/run_eval.py --model Qwen/Qwen3.5-4B \
    --envs glyphbench/minigrid-doorkey-6x6-v0 glyphbench/atari-pong-v0
```

Results include per-env metrics (mean return, episode length, parse-failure rate) plus full trajectory recordings (observation, LLM response, action, reward at every step) for replay verification.

## Harness modes

Each model is evaluated in four conditions:

| Mode | Context | Response |
|---|---|---|
| `markov_zeroshot`  | single observation | just an action |
| `markov_cot`       | single observation | thinking + action |
| `history_zeroshot` | last N obs + actions | just an action |
| `history_cot`      | last N obs + actions | thinking + action |

Comparing `markov` vs `history` shows which envs need memory; comparing `zeroshot` vs `cot` shows which need reasoning.

## Scoring

**GlyphBench Score** = equal-weight mean of per-suite interquartile means, after normalising each env against a fixed-seed random baseline ([`eval/random_baseline.json`](eval/random_baseline.json)).

Per-env normalisation: `(model_return − random_return) / max(|random_return|, 1)`, clipped to `[−1, 10]`. We report raw `mean_return` alongside the normalised score — no opaque success rates.

## Trajectory replay and GIFs

```bash
# Replay a recorded trajectory with color
uv run python scripts/replay_trajectory.py results/Qwen_Qwen3.5-4B/history_cot/trajectories/glyphbench__atari-pong-v0/seed_42_ep_0.jsonl

# Export a single trajectory as a GIF
uv run python scripts/replay_trajectory.py trajectory.jsonl --gif output.gif

# Render a random-agent GIF for every env (seeds 42, natural termination)
uv run python scripts/record_random_gifs.py --output docs/leaderboard/gifs/
```

## Interactive demo

```bash
# Watch a random agent play a short clip of each env
uv run python scripts/demo_all_envs.py --steps 10 --delay 0.3 --pause

# Single env, slower playback
uv run python scripts/demo_all_envs.py --env glyphbench/craftax-classic-v0 --delay 0.2
```

The renderer is flicker-free (single-write frame with ANSI cursor-home).

## Using GlyphBench with your own agent

```python
import glyphbench
import gymnasium as gym

env = gym.make("glyphbench/minigrid-doorkey-6x6-v0")
obs, info = env.reset(seed=42)

done, total = False, 0.0
while not done:
    action = your_agent(obs, env.unwrapped.action_spec.names)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated

print(f"Episode return: {total}")
```

## Project layout

```
src/glyphbench/
    core/                # BaseAsciiEnv, GridObservation, ActionSpec, registry
    envs/                # 6 suites · 292 envs
    harness/             # LLM agent loop, prompt builder, JSON parser
    providers/           # vLLM, OpenAI, Anthropic, Gemini clients
    runner/              # Async benchmark runner, config, dashboard, storage
eval/                    # Batched vLLM runner, random baseline, scoring
cluster_manager/         # SLURM multi-cluster experiment manager
scripts/                 # Demo, trajectory replay, GIF export, upload tools
docs/leaderboard/        # GitHub Pages site (leaderboard + rollout gallery)
```

## Development

```bash
uv sync --all-extras
uv run pytest                          # 2108+ tests
uv run ruff check src/
uv run mypy src/glyphbench/
```

## Citation

```bibtex
@article{glyphbench2026,
  title   = {GlyphBench: A Unified Benchmark for Evaluating LLM Agents on Sequential Decision-Making},
  author  = {Anonymous},
  year    = {2026},
}
```

## License

MIT
