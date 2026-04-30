# GlyphBench

A benchmark of **292 text-rendered reinforcement-learning environments** for evaluating LLM agents on sequential decision-making.

Every environment renders its state as a Unicode text grid with a legend and discrete named actions. The agent sees only the grid — no privileged state-channel — so every game-relevant fact must be readable off the glyphs themselves. Observations are deterministic (seeded), making results fully reproducible.

- **Leaderboard & rollouts:** [roger-creus.github.io/glyphbench](https://roger-creus.github.io/glyphbench/leaderboard/)
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
uv add "glyphbench[eval]"            # + verifiers + vLLM (eval + RL integration)
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
import glyphbench
from glyphbench.core import make_env

# Direct game loop
env = make_env("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(42)
print(obs)

# Or: load as a verifiers environment for eval / RL
vf_env = glyphbench.load_environment(task_id="glyphbench/minigrid-empty-5x5-v0")
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

Every environment also exposes `env.system_prompt()` — a compact description of rules, actions, reward structure, and termination conditions, ready to pass as a system message to any LLM.

## Observation format

Every environment returns a single text string. The harness shows the model:

- `[Legend]` — maps each glyph to its meaning (rendered once, deduped).
- `[Grid]` — the 2D Unicode grid (the only required channel).
- `[Message]` — optional per-turn narrative event ("You bumped a wall.").
- `[Actions]` — the action vocabulary the model must pick from this turn.

Envs may compute a `[HUD]` (HP, inventory, score, etc.) for their `info` dict
and trajectory logs, but the harness deliberately does not show it to the
model. Privileged state (mob positions, hidden inventory) must therefore be
encoded in the visible grid for the agent to reason about.

## Running LLM evaluations

GlyphBench registers every environment as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
environment, so any OpenAI-compatible endpoint can be evaluated via `vf-eval`.
Two turnkey wrappers are provided:

```bash
# Short smoke (1 env, 1 episode) — useful for wiring checks
bash eval/run_debug.sh

# Full sweep (all 292 envs, configurable episodes via $EPISODES / $MODEL)
bash eval/run_full.sh
```

Both scripts assume an OpenAI-compatible server is reachable at `http://localhost:8000/v1`
(e.g. `uv run vllm serve Qwen/Qwen3.5-4B --port 8000`). See [`eval/README.md`](eval/README.md)
for full arguments.

At a Python level, the single entry point is:

```python
import glyphbench
env = glyphbench.load_environment(
    task_id="glyphbench/minigrid-empty-5x5-v0",
    num_episodes=5,              # default
    n_frames=0,                  # stateless per turn (default)
    max_output_tokens=8192,      # match your --max-tokens
    use_memory=False,            # optional carried memory scaffold
)
```

which returns a `verifiers.MultiTurnEnv` ready for `vf.evaluate(...)` or RL training.

## Training (RL fine-tuning with prime-rl)

GlyphBench ships an RL training pipeline that fine-tunes Qwen-class models
on all 292 envs under the same inference profile we eval at (thinking on,
8K action + 4K memory, memory mode).

```bash
# 1. Install the rl extra
uv sync --extra rl --extra eval

# 2. Set up cluster config (one-time per cluster)
cp .env.cluster.template .env.cluster
$EDITOR .env.cluster   # fill in node names, ports, API keys, output dir

# 3. Launch all components from the trainer node
bash scripts/rl/launch_all.sh
```

See:

- `configs/rl/qwen35-4b-glyphbench/README.md` — config-specific notes
- `scripts/rl/README.md` — operator guide
- `src/glyphbench/rl/README.md` — design notes for the custom advantage and
  loss hooks

## Harness

One harness mode: per turn the model sees only `[system, current observation]`
— no HUD side-channel, every game-relevant fact must be readable off the
Unicode grid. An optional frame-stacked history window is available
(`n_frames=N`); the default is `n_frames=0` (stateless / pure Markov). The
system prompt advertises the output-token budget the eval is run with so the
model can self-pace its reasoning. All observations are deterministic —
identical seeds produce identical trajectories.

Set `use_memory=True` to add an opt-in memory scaffold across all environments.
Each environment step then uses two model generations: one for the action and
one for a concise memory update conditioned on the action response, reward,
done flags, and the same HUD-stripped next-observation view used for action
selection. For RL, those two generations are stored as one trajectory step so
action tokens and memory-update tokens train together with the same task reward.
`memory_update_max_tokens` can override only the second generation's token
limit; by default it reuses the action sampling limit.
Memory-aware trajectories show previous and updated memory in `glyphbench replay`;
the standalone trajectory/GIF renderer also includes stored memory when present.

## Scoring

GlyphBench reports **raw episodic return per (env, model)**. There is no
benchmark-wide normalised score: we publish the raw per-task per-model means
and let downstream analyses choose their own aggregation. A reproducible
random-agent baseline is available at
[`eval/random_baseline.json`](eval/random_baseline.json) for callers that
want a zero-skill reference.

## Trajectory replay and GIFs

```bash
# Replay every saved trajectory under a results directory (rich TUI)
uv run glyphbench replay cluster_manager/results --suite minigrid --pause

# Replay a single recorded trajectory with color
uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl

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
from glyphbench.core import make_env

env = make_env("glyphbench/minigrid-doorkey-6x6-v0")
obs, info = env.reset(42)

done, total = False, 0.0
while not done:
    action = your_agent(obs, env.action_spec.names)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated

print(f"Episode return: {total}")
```

## Project layout

```
src/glyphbench/
    core/                  # BaseGlyphEnv, GridObservation, ActionSpec, registry
    envs/                  # 6 suites · 292 envs
    verifiers_integration/ # prompt builder, parser, multi-turn env, rubric
    plotting/              # parquet loaders + paper-figure generators
eval/                      # vf-eval wrappers, random baseline
configs/                   # endpoint registry, prime-rl training configs
cluster_manager/           # SLURM multi-cluster experiment manager
scripts/                   # Demo, trajectory replay, GIF export, upload tools
docs/leaderboard/          # GitHub Pages site (leaderboard + rollout gallery)
```

## Development

```bash
uv sync --all-extras
uv run pytest
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
