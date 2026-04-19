# atlas_rl.core

Core contracts. Every env, harness component, and runner depends on this package. This package depends on **nothing internal** — no imports from `envs/`, `harness/`, `runner/`, or `providers/`.

## Public types

- `GridObservation` — frozen dataclass with `grid`, `legend`, `hud`, `message` string fields. Its `render()` method produces the canonical prompt-ready string.
- `ActionSpec` — frozen dataclass with `names`, `descriptions`. The env's action vocabulary is defined once and never per-turn filtered.
- `BaseAsciiEnv` — abstract gymnasium env. Subclasses implement `_reset`, `_step`, `_render_current_observation`, `system_prompt`, `env_id`. Reset requires an explicit seed.

## Invariants

1. Every observation has all four string fields populated (empty string is fine).
2. Every env requires an explicit seed on `reset`.
3. All randomness must go through `self.rng` (a `numpy.random.Generator`).
4. This package imports nothing from the rest of the codebase.

See `specs/2026-04-12-stage-0-foundation-and-pilot.md` §3 for the full contract.
