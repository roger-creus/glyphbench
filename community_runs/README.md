# Community Runs

This directory holds externally-submitted eval runs that feed the [GlyphBench leaderboard](https://roger-creus.github.io/glyphbench/leaderboard/).

## Submitting a run

1. Produce results via `eval/run_eval.py` (see `../CONTRIBUTING.md` for the official protocol).
2. Add a directory here named `<author>__<model-id-slug>__<harness>/` containing:
   - `results.json` — aggregate metrics (required)
   - `per_env/*.json` — per-environment metrics (required)
   - `meta.json` — `{"author": "...", "model": "...", "harness": "...", "seed_set": "official", "commit": "<sha>", "date": "YYYY-MM-DD"}`
   - **Do not** commit the full trajectory `.jsonl` files to git — upload them as a GitHub Release asset and link from `meta.json` via `"trajectories_url"`.
3. Open a PR. CI will validate the submission and spot-check trajectories.

Submissions that change the official protocol (different `episodes`, different `max_turns`, different seeds) live in a separate `unofficial/` subdirectory and are displayed on the leaderboard in a clearly-labeled "unofficial" tab — they are valuable data but cannot be directly compared with official runs.
