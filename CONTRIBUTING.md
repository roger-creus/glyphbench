# Contributing to GlyphBench

Welcome! GlyphBench is an open benchmark — we want more environments, more models evaluated, and better tooling. This guide covers the three main ways to contribute.

## Table of Contents

1. [Submitting a model to the leaderboard](#submitting-a-model-to-the-leaderboard)
2. [Adding a new environment](#adding-a-new-environment)
3. [Adding a new LLM provider](#adding-a-new-llm-provider)
4. [Code style & test requirements](#code-style--test-requirements)

## Submitting a model to the leaderboard

Results on the leaderboard must be reproducible. A valid submission includes:

1. **Results artifact** — `results.json` plus `per_env/*.json` produced by `eval/run_eval.py`
2. **Trajectories** — the full `.jsonl` trajectory files under `trajectories/` (used for anti-cheat replay verification)
3. **Meta** — model id, harness mode, seed, commit SHA of this repo, date

**How to submit:**

```bash
# 1. Run the official eval (25 episodes, all 300 envs, each env's natural step budget)
uv run python eval/run_eval.py \
    --model <your-model-id> \
    --episodes 25 \
    --harness history_cot \
    --output results/<your-model-id>__history_cot/

# 2. Zip the run directory (including trajectories)
tar czf run.tar.gz results/<your-model-id>__history_cot/

# 3. Open a PR against github.com/roger-creus/glyphbench
#    - Add your run under `community_runs/<your-model-id>__<harness>/`
#    - Include results.json and per_env/*.json
#    - Upload the trajectory archive as a release asset (we link it from the PR)
```

CI will:
- Validate the schema of `results.json` and each `per_env/*.json`
- Spot-check 3 random trajectories: replay each one to confirm the recorded actions reproduce the recorded rewards
- Reject submissions where seeds or the step budget deviate from the official protocol

The leaderboard auto-updates nightly from merged PRs.

### Official protocol

- **episodes:** 25 per environment
- **max_turns:** each env's natural step budget (do not pass `--max-turns` to override)
- **harness modes:** at least one of `markov_zeroshot` / `markov_cot` / `history_zeroshot` / `history_cot`
- **seeds:** fixed via `eval/run_eval.py` seed-derivation (do not override)
- **temperature:** 0.7 (default; specify if different)
- **max_new_tokens:** 16384 for thinking-enabled runs

## Adding a new environment

1. Create a new file in `src/glyphbench/envs/<suite>/<env_name>.py`
2. Extend the suite's base class (e.g., `MiniGridBase`, `MiniHackBase`, `ProcgenBase`, `AtariBase`) or subclass `BaseAsciiEnv` directly
3. Implement required methods:
   - `env_id()` — return the canonical `glyphbench/<suite>-<name>-v0` id
   - `_reset(seed)` — return initial `GridObservation`
   - `_step(action)` — return `(obs, reward, terminated, truncated, info)`
   - `system_prompt()` — full rules, actions, reward structure
4. Register in `src/glyphbench/envs/<suite>/__init__.py` via `register_env(...)`
5. Add tests in `tests/envs/<suite>/test_<env_name>.py`
6. Update `eval/random_baseline.json` by running `uv run python eval/random_baseline.py --episodes 25 --suites <suite>`

### Design rules for new environments

- **Discriminative signal.** A worse model must score lower than a better one. If an env is too opaque or too trivial, cut it.
- **Complete system prompt.** All rules, all actions, the reward structure. No hidden gotchas.
- **Unicode glyphs.** Single Unicode codepoints per cell, no symbol collisions with other envs in the suite. See `src/glyphbench/envs/<suite>/` for conventions.
- **Invalid actions = NOOP**, never a reward penalty.
- **Deterministic seeding.** Same seed = same trajectory.
- **Signal range.** Random agent should score near zero; optimal play should be clearly higher.

### Test requirements

Every environment must pass:
- Seed determinism (same seed produces same trajectory)
- Observation contract (text obs, valid structure, renderable legend + grid)
- Action space validation (int in `[0, n_actions)`)
- Random rollout without crashes (200+ steps across 10 seeds)
- Suite conformance test (auto-validated for the suite)

## Adding a new LLM provider

1. Create `src/glyphbench/providers/<provider>_client.py`
2. Implement the `LLMClient` protocol (see `providers/base.py`)
3. Add to `providers/factory.py`
4. Add pricing to `pricing.yaml` (input-token price / output-token price, USD/1M tokens)

## Code style & test requirements

- Python 3.11+
- Ruff for linting/formatting (line length 100)
- Mypy strict mode for all new code in `src/glyphbench/`
- TDD: write the failing test first, then the minimal implementation

```bash
uv run pytest                              # full suite (2255+ tests)
uv run pytest tests/envs/minigrid/ -v      # single suite
uv run ruff check src/ tests/              # lint
uv run mypy src/glyphbench/                # type check
```

## Local development

```bash
git clone https://github.com/roger-creus/glyphbench.git
cd glyphbench
uv sync --all-extras
uv run pytest
```

## Questions?

Open a GitHub Discussion or file an issue. We aim to respond within a few days.
