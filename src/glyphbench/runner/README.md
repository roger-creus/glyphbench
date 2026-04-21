# glyphbench.runner

Async parallel-episode runner over a set of envs × seeds × episodes. Backs into the harness, provider clients, and envs. Writes JSONL + parquet to `runs/<run_id>/`.

## Public types

- `RunConfig` — Pydantic model for the YAML run config.
- `run_benchmark(config)` — the async entry point.
- `RunStorage` — per-run output directory manager.
- `CostTracker` — shared cost accumulator with hard abort on budget overflow.
- `Dashboard` — rich live dashboard (and `NullDashboard` no-op).
- `RandomAgent` — uniform-random-action baseline; same interface as `HarnessAgent`.

## Concurrency model

Episodes are the unit of parallelism. A `Semaphore(config.concurrency)` bounds how many episodes run simultaneously. Turns within one episode are strictly sequential (LLM → env step → LLM).

## Invariants

1. Each (env, seed, episode_idx) gets a derived episode seed via stable SHA256 hash.
2. `budget_usd=None` means unlimited; any other value aborts the run on overflow.
3. The runner fails loud (`return_exceptions=False`) on any episode error in Stage 0; later stages may relax this.
4. JSONL trajectories are opt-in (`harness.trajectory_logging=true`); parquet summaries are always written.
