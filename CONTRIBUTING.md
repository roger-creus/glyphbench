# Contributing to GlyphBench

## Adding a New Environment

1. Create a new file in `src/glyphbench/envs/<suite>/<env_name>.py`
2. Extend the suite's base class (e.g., `MiniGridBase`, `MiniHackBase`, `ProcgenBase`, `AtariBase`)
3. Implement required methods: `env_id()`, `_generate_grid/level(seed)`, game step logic
4. Register in `src/glyphbench/envs/<suite>/__init__.py`
5. Add tests in `tests/envs/<suite>/test_<env_name>.py`
6. Add a system prompt template if using Jinja2 templates

### Test Requirements

Every environment must pass:
- Seed determinism (same seed = same trajectory)
- Observation contract (string obs, valid structure)
- Action space validation
- Random rollout without crashes (200+ steps)
- Suite conformance test (auto-validated)

## Adding a New LLM Provider

1. Create `src/glyphbench/providers/<provider>_client.py`
2. Implement the `LLMClient` protocol (see `providers/base.py`)
3. Add to `providers/factory.py`
4. Add pricing to `pricing.yaml`

## Code Style

- Python 3.11+
- Ruff for linting/formatting (line length 100)
- Mypy strict mode
- TDD: write tests first

## Running Tests

```bash
uv run pytest                    # full suite
uv run pytest tests/envs/minigrid/ -v  # single suite
uv run ruff check src/           # lint
uv run mypy src/glyphbench/        # type check
```
