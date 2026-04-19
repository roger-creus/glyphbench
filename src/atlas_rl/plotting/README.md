# atlas_rl.plotting

Paper-ready plotting utilities for benchmark results.

## Architecture

This module depends **only** on the on-disk parquet format produced by `RunStorage`.
It does NOT import from `runner`, `providers`, `harness`, or `envs`.

## Modules

### style.py

Matplotlib rcParams for NeurIPS-style figures.

```python
from atlas_rl.plotting.style import apply_style, paper_style

apply_style()           # set globally
with paper_style():     # scoped context manager
    plt.plot(...)
```

Key settings: Computer Modern serif, 300 DPI save, no top/right spines, subtle grid.

### common.py

Data loaders for the parquet output format.

```python
from atlas_rl.plotting.common import load_run, load_runs, compute_normalized_scores

data = load_run("runs/my-run")           # {'summary': DataFrame, 'turns': DataFrame}
combined = load_runs(["runs/r1", "runs/r2"])  # concatenated summaries with run_id column
normalized = compute_normalized_scores(summary, random_summary, expert_reference={"env-a": 20.0})
```

### notebooks/foundation_smoke.py

Standalone script producing 7 paper-ready plots:

1. Return distribution (violin)
2. Episode length (box)
3. Token usage (grouped bar)
4. Parse failure rate (bar)
5. Action distribution (stacked bar, top-10)
6. Cost per episode (bar)
7. Return over episodes (line)

```bash
uv run python -m atlas_rl.plotting.notebooks.foundation_smoke runs/<run_id>
```

Saves to `runs/<run_id>/figures/*.{pdf,png}`.

## Contract

- All functions accept string paths, not Path objects, for CLI ergonomics.
- `load_run` returns a dict with exactly two keys: `summary` and `turns`.
- `compute_normalized_scores` mutates nothing; returns a new DataFrame.
- Plots use `PAPER_STYLE` via the `paper_style()` context manager.
