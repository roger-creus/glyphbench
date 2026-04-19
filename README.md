# ATLAS - ASCII Testbed for LLM Agent Skills

A unified ASCII-based benchmark of 169 discrete-action RL environments for evaluating LLMs on sequential decision-making.

## Suites

| Suite | Envs | Description |
|-------|------|-------------|
| MiniGrid | 71 | Grid navigation, key/door puzzles, dynamic obstacles |
| MiniHack | 63 | NetHack-inspired dungeon crawling, combat, items, skills |
| Procgen | 16 | Procedurally generated action games (platformers, shooters, mazes) |
| Craftax | 1 | Open-world survival crafting with 22 achievements |
| Atari | 17 | Classic arcade games (Pac-Man, Space Invaders, Breakout, etc.) |

## Quickstart

```bash
# Install
git clone <repo-url>
cd rl-world-ascii
uv sync --all-extras

# Run tests
uv run pytest

# Run a benchmark (requires a vLLM server)
uv run python scripts/run_benchmark.py configs/examples/foundation_smoke.yaml

# Generate plots from results
uv run python -m atlas_rl.plotting.notebooks.foundation_smoke runs/<run_id>
```

## Supported LLM Providers

- **vLLM** (local, open-source models) - recommended for development
- **OpenAI** (GPT-4o, o1, o1-mini)
- **Anthropic** (Claude Opus / Sonnet / Haiku)
- **Google** (Gemini 2.0 Pro / Flash)

## Architecture

Every environment implements the same interface:
- ASCII grid observation with legend and HUD
- Discrete action space with named actions
- Deterministic (seeded) episode replay
- JSON-based LLM response parsing with retry/fallback

The benchmark runner orchestrates parallel episodes, tracks costs, and produces parquet output for analysis.

## Project Structure

```
src/atlas_rl/
    core/       # Base env, observation, action, metrics, registry
    harness/    # LLM agent, prompt builder, parser, system prompts
    providers/  # vLLM, OpenAI, Anthropic, Gemini clients
    runner/     # Benchmark runner, config, dashboard, storage
    plotting/   # Paper-ready figures, data loaders, normalization
    envs/
        minigrid/   # 71 MiniGrid environments
        minihack/   # 63 MiniHack environments
        procgen/    # 16 Procgen games
        craftax/    # Craftax Classic (22 achievements)
        atari/      # 17 Atari games
```

## Citation

```bibtex
@article{atlas2026,
  title={ATLAS: A Unified ASCII Testbed for LLM Agent Skills},
  year={2026},
}
```

## License

MIT
