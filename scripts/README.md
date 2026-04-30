# scripts/

Top-level operator scripts. Each script's `--help` (or its docstring header)
has the full flag list; this index just shows the canonical invocation.

## Demo + replay

- **`demo_all_envs.py`** — Watch a uniform-random agent play any env in the
  same rich TUI layout `gb replay` uses (header bar, system-prompt panel,
  grid + side panels). Useful for smoke-testing a freshly installed env or
  visually scanning an entire suite.
  ```bash
  uv run python scripts/demo_all_envs.py --suite minigrid --delay 0.1
  uv run python scripts/demo_all_envs.py --env glyphbench/craftax-classic-v0 --pause
  uv run python scripts/demo_all_envs.py --list
  ```
  Pause-mode hotkeys: `→` advance, `←` rewind, `s` system prompt, `l` legend,
  `a` action list, `q`/`n` next env.

- **`replay_trajectory.py`** — Replay a single saved trajectory `.jsonl` with
  Unicode + color in the terminal, or export it as a GIF.
  ```bash
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --delay 0.2
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --gif out.gif
  uv run python scripts/replay_trajectory.py path/to/trajectories/   # replay all in dir
  ```
  Note: this is the legacy single-file viewer. For browsing whole runs
  directories with multi-panel TUI and pause-mode hotkeys, use `gb replay`
  (see `docs/REPLAY.md`).

## Asset generation

- **`record_random_gifs.py`** — Render a random-agent GIF for every env (or a
  subset) via the `replay_trajectory.export_gif` renderer. The publicly hosted
  GIFs on HF are produced by a separate pipeline; this script is for callers
  who want their own local GIFs.
  ```bash
  uv run python scripts/record_random_gifs.py --output ./out/gifs/
  uv run python scripts/record_random_gifs.py --suite minigrid --steps 30
  uv run python scripts/record_random_gifs.py --env glyphbench/atari-pong-v0
  ```

- **`upload_assets.py`** — Upload a directory of files to the GlyphBench HF
  dataset repo. Used to push regenerated GIFs to
  `anon-paper-submission/glyphbench-assets`.
  ```bash
  uv run python scripts/upload_assets.py --src docs/leaderboard/gifs --dst gifs
  uv run python scripts/upload_assets.py --repo owner/name --private
  ```

## Leaderboard / catalog

- **`build_leaderboard.py`** — Aggregate `results/` run directories into
  `docs/leaderboard/data.json`, which is consumed by the GitHub Pages
  leaderboard frontend.
  ```bash
  uv run python scripts/build_leaderboard.py --results results/ --output docs/leaderboard/data.json
  ```

- **`generate_env_catalog.py`** — Refresh `docs/ENVIRONMENTS.md` from the
  current env registry (derives suite groupings and total env count
  automatically).
  ```bash
  uv run python scripts/generate_env_catalog.py
  ```

## Manual play / debugging

- **`play_random.py`** — Play any single env with random actions, printing
  each step to stdout. The simplest possible sanity check for a new env.
  ```bash
  uv run python scripts/play_random.py glyphbench/craftax-classic-v0
  uv run python scripts/play_random.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42 --steps 50
  uv run python scripts/play_random.py glyphbench/minihack-eat-v0 --delay 0.3
  ```

- **`play_curses.py`** — Curses-based random-agent viewer: renders the grid
  with colors, a side panel for HUD/legend, and a bottom bar for messages
  and reward tracking.
  ```bash
  uv run python scripts/play_curses.py glyphbench/craftax-classic-v0
  uv run python scripts/play_curses.py glyphbench/atari-pong-v0 --delay 0.05
  ```

- **`play_interactive.py`** — Interactive curses player: shows numbered
  actions; press the corresponding number key to act. Good for manually
  exploring episode dynamics.
  ```bash
  uv run python scripts/play_interactive.py glyphbench/craftax-classic-v0
  uv run python scripts/play_interactive.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42
  ```

## Misc

- **`terminal_colors.py`** — Curses color mapping library (not a standalone
  script). Provides `char_attr(ch)` and `init_colors()` used by
  `play_curses.py` and `play_interactive.py`. Import directly; has no
  `__main__` entry point.

- **`build_sif.sh`** — Build the GlyphBench Docker image and convert it to
  a Singularity/Apptainer SIF file. Respects `IMAGE` and `SIF` env vars.
  ```bash
  IMAGE=glyphbench:latest SIF=glyphbench.sif bash scripts/build_sif.sh
  ```
