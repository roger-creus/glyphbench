# Architecture

## Directory layout

```
src/glyphbench/
    core/                   # BaseGlyphEnv, GridObservation, ActionSpec, registry
    envs/                   # 6 suites · 300 envs (atari, classics, craftax,
                            #   minigrid, minihack, procgen)
    envs/craftax/docs/      # 10-chapter LLM-first tutorial composed into
                            #   craftax env system prompts
    verifiers_integration/  # prompt builder, parser, multi-turn env, rubric
    plotting/               # parquet loaders + paper-figure generators
    rl/                     # custom advantage / loss hooks for prime-rl
    cli.py                  # `glyphbench` / `gb replay` CLI entry point

eval/                       # vf-eval wrappers, random-agent baseline
configs/                    # endpoint registry (endpoints.toml)
scripts/                    # demo, trajectory replay, GIF export, upload tools
docs/                       # this directory + REPLAY, OBSERVATION_FORMAT,
                            #   ENVIRONMENTS, INTEGRATION, leaderboard site
docs/leaderboard/           # GitHub Pages site (leaderboard + rollout gallery)
```

## Key boundaries

- **Envs declare `system_prompt()`, `reset()`, `step()`** and emit text observations.
  They never know about the LLM or the harness — just pure simulators.
- **The harness** (`verifiers_integration/`) composes the system prompt, parses
  the model's action XML, runs `env.step`, and packages everything for the
  verifiers `MultiTurnEnv` API.
- **`cli.py`** (the `gb replay` TUI) reads verifiers `results.jsonl` files and
  renders them with Rich. Lazily instantiates one env per `env_id` for the
  parser canonicalisation pass.
- **`rl/`** plugs into prime-rl as a custom advantage + loss; per-env Welford
  tracks reward statistics for normalization. See `src/glyphbench/rl/README.md`.

## Suite catalog

Per-suite env counts (from the registry):

| Suite | Envs | Action space |
|---|---:|---|
| MiniGrid | 71 | 7 actions |
| MiniHack | 63 | 22 actions |
| Atari | 57 | 3–10 actions |
| Classics | 50 | 2–256 actions (puzzle envs have very large discrete spaces) |
| Craftax | 43 | 19 / 45 actions (classic / full) |
| Procgen | 16 | 4–6 actions |
| **Total** | **300** | |
