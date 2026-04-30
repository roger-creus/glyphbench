# README public-release shine — design

**Date:** 2026-04-30
**Scope:** Main `README.md` rewrite + per-area subdir docs + four polished GIF assets, ahead of making the repo public.
**Status:** approved by user, ready for plan

## Context

The repo is about to be made public. The current `README.md` (259 lines) is functional but plain — no images, no badges, no GIFs — and several factual claims in it are stale relative to the codebase (memory-step split, env counts, RL pipeline state, etc.). At the same time the project ships a rich rendering layer (the `gb replay` TUI and `demo_all_envs.py` both render through a shared `_build_frame` producing a multi-panel Rich layout), and 292 random-agent GIFs already exist on disk — but those GIFs were produced by an older `replay_trajectory.export_gif` whose color map only knows ASCII tokens (`#`, `^`, `>`), while the actual observations stream raw Unicode glyphs (`█`, `→`, `★`). The result: the existing GIFs render every cell in default gray and don't match what a user actually sees in the live tools.

This spec ships:
1. A **compact, shiny main README** (target ~150 lines) with hero GIF, gallery table, badges, two embedded tool GIFs, and a documentation map that funnels readers to per-area docs.
2. A set of **per-area subdir docs** that absorb the depth currently bloating the main README.
3. Four polished asset categories produced by a brand-new throwaway pipeline that uses the modern renderer (no drift from the live tools), with per-glyph color baked in so the gallery looks "beautiful and colorful".
4. A **README freshness audit** that catches every stale claim before the rewrite.
5. Drop-in replacement of the existing 292 HF-hosted GIFs with all 300, regenerated through the new pipeline.

Throwaway code is gitignored under `tools/readme-assets/` and removed at the end. The only persistent changes are the README, the new/refreshed subdir docs, and the HF assets.

## Decisions (locked during brainstorm)

1. **Hero strategy = composite GIF + per-suite gallery table.** A wide composite "sizzle reel" at the top, plus a 4×3 markdown table further down with one representative GIF per cell.
2. **Gallery density = 2 envs per suite, 12 GIFs total.** Below that, one inline link to the existing leaderboard `gallery.html` for the full 300.
3. **GIF hosting = the existing HF dataset repo `anon-paper-submission/glyphbench-assets`.** Repo name kept as-is for now (anonymization can be revisited later); README links to HF raw URLs so the repo carries no GIF binaries.
4. **`gb replay` GIF style = pause-mode walkthrough with hotkey hops** (`→ → s [pager-q] r [pager-q] m [pager-q] q`). The pager popping out and being dismissed is the visual that demonstrates the hotkey-driven pager feature. README caption underneath the GIF lists the keys (`s` system / `r` reasoning / `m` memory / `←/→` step / `q` next).
5. **Top-of-README polish = centered hero block + status badges + quick-nav strip**, no custom SVG/PNG banner.
6. **Code-quality scope = the new asset code only.** No CI, no repo-wide lint/type/test pass. Existing `record_random_gifs.py` / `replay_trajectory.py` / `upload_assets.py` are not modified.
7. **Color = per-glyph palette (player=green, walls=gray, goal=gold, water=blue, monsters=red, items=magenta, …)** baked into the gallery/hero GIFs. The full-TUI GIFs (replay + demo) capture the actual TUI verbatim via asciinema+agg, which is white-on-dark.
8. **Two distinct rendering pipelines:**
   - **Grid-only** (gallery + hero) — pure Python + Pillow + per-glyph color palette, headless, fast.
   - **Full-TUI** (replay + demo) — asciinema rec → agg → GIF, captures the real Rich-rendered tool.
9. **Validation gate before mass regeneration:** render 3 sample envs only, show user, get explicit OK before running on all 300.
10. **README split = compact hub + per-area subdir docs.** Main README stays ~150 lines; depth moves into `docs/OBSERVATION_FORMAT.md`, `docs/INTEGRATION.md`, `docs/ARCHITECTURE.md`, `scripts/README.md`, `eval/README.md`, `src/glyphbench/rl/README.md`, `docs/REPLAY.md`.
11. **README freshness audit precedes the rewrite.** A discrepancy list (`tools/readme-assets/audit.md`, gitignored) drives both the main README rewrite and each subdir doc.
12. **Anonymization is not auto-rewritten.** Strings like `anon-paper-submission`, `Anonymous` author, etc. are flagged in the audit for the user's call.

## Asset deliverables

Four asset categories, all hosted on HF dataset `anon-paper-submission/glyphbench-assets`:

| # | Asset | Path on HF | Mode | Source |
|---|---|---|---|---|
| 1 | **300 grid-only GIFs** | `gifs/glyphbench__<env_id>.gif` | Colored Unicode `[Grid]` panel + 1-line header (`env_id · turn N/T · action · return`). ~600px wide. Per-glyph color palette baked in. | `tools/readme-assets/render_grid_gif.py` (new throwaway) |
| 2 | **Hero composite** | `readme/hero.gif` | 12 of the above ffmpeg-stitched with crossfades + per-cell suite caption. ~900px wide, ~12s loop. | ffmpeg-concat in `tools/readme-assets/build_hero.sh` |
| 3 | **demo_all_envs walkthrough** | `readme/demo_all_envs.gif` | Full TUI in continuous mode, picks `craftax-classic-v0` so all panels populate (HUD, legend, recent actions, env feedback). | asciinema rec of `demo_all_envs.py --env craftax-classic-v0 --delay 0.1` → agg |
| 4 | **gb replay walkthrough** | `readme/gb_replay.gif` | Full TUI in pause mode with hotkey hops (`→ → s [pager-q] r [pager-q] m [pager-q] q`). Targets `minigrid-doorkey-6x6-v0` (different env from #3 to avoid visual repeat; smaller grid leaves more space for side panels). | Hand-driven asciinema rec of `glyphbench replay <runs_dir> --pause --env glyphbench/minigrid-doorkey-6x6-v0` → agg |

Drop-in replacement of the 292 existing HF GIFs by category #1 means the leaderboard `gallery.html` automatically inherits the better-looking renders.

## Glyph color palette

Hand-curated table in `tools/readme-assets/render_grid_gif.py`. Keys are Unicode codepoints, values are RGB tuples. Coverage targets ~25 common glyphs; anything unmapped renders default light gray (won't break, just blends into walls).

Initial palette (subject to per-suite tuning during the validation gate):

| Glyph | Meaning | Color |
|---|---|---|
| `→` `↓` `←` `↑` `@` `☺` | Player / agent | bold green |
| `█` | Wall | dark gray |
| `·` ` ` | Floor | very dark gray |
| `★` `*` `✶` | Goal / star | gold |
| `≈` `~` | Water | blue |
| `▣` `D` | Door | yellow |
| `🔑` `K` | Key | magenta |
| `✚` `+` | Health / pickup | bright cyan |
| `✖` `X` `!` `⚠` | Hazard | red |
| `○` `●` | Mob / enemy | white |
| `▲` `▼` | Stairs / level | dim cyan |

Final palette is reviewed in the validation gate (sample 3 envs render).

## Main README structure (target ~150 lines)

```
1. Hero block (centered)
   - <h1> GlyphBench
   - Tagline (one line, current: "A benchmark of 300 text-rendered RL environments…")
   - hero.gif (HF raw URL)
   - Quick-nav: [ Leaderboard · Paper · Quickstart · Contributing ]
   - Badges: License MIT · Python 3.10+ · vLLM · Verifiers · Leaderboard

2. At a glance (existing 6-row suite table — preserved + counts re-verified)

3. Browse the suites
   - 4×3 markdown table of 12 chosen GIFs with env-name captions
   - One-line: "→ See all 300 environments at <leaderboard gallery link>"

4. Install (3 lines, current commands re-verified against pyproject)

5. Quickstart (one ~10-line Python snippet, API re-verified)

6. Tools
   - Replay  — gb_replay.gif + 1-line caption + link to docs/REPLAY.md
   - Demo    — demo_all_envs.gif + 1-line caption + link to scripts/README.md

7. Documentation map (8 bullets, see below)

8. Citation + License
```

Documentation map at the bottom:

```markdown
- Observation format · harness — docs/OBSERVATION_FORMAT.md
- LLM evaluation (vLLM / verifiers) — eval/README.md
- RL training (prime-rl) — src/glyphbench/rl/README.md
- Trajectory replay tool — docs/REPLAY.md
- Interactive demo & scripts — scripts/README.md
- Architecture — docs/ARCHITECTURE.md
- Use with your own agent — docs/INTEGRATION.md
- Contributing — CONTRIBUTING.md
```

## The 12 hero/gallery envs

Two per suite, picked for iconicity, reasonable random-agent rollout length, and visual contrast:

| Suite | Env A | Env B |
|---|---|---|
| MiniGrid | `minigrid-doorkey-6x6-v0` | `minigrid-multiroom-n4-s5-v0` |
| MiniHack | `minihack-room-monster-15x15-v0` | `minihack-quest-easy-v0` |
| Atari | `atari-pong-v0` | `atari-breakout-v0` |
| Classics | `classics-snake-medium-v0` | `classics-sokoban-easy-v0` |
| Craftax | `craftax-classic-v0` | `craftax-fight-cow-v0` |
| Procgen | `procgen-coinrun-v0` | `procgen-maze-v0` |

Picks are subject to swap during the validation gate if any chosen env renders weakly.

## Per-area subdir docs

| File | Status | Job |
|---|---|---|
| `docs/OBSERVATION_FORMAT.md` | new | Observation format + harness behavior + frame-stacking + memory mode in one place |
| `docs/INTEGRATION.md` | new | "Use it with your own agent" cookbook + trajectory inspection tips |
| `docs/ARCHITECTURE.md` | new, brief | Project layout tree + 1-line per directory |
| `scripts/README.md` | new | Index of all top-level scripts (demo, replay, GIF export, baseline, upload, build_leaderboard, generate_env_catalog) with one-liner usage each |
| `docs/REPLAY.md` | refresh + embed `gb_replay.gif` | Already strong; verify currency + add the GIF at the top |
| `eval/README.md` | refresh | Absorb scoring detail + vf-eval flag enumeration from current main README |
| `src/glyphbench/rl/README.md` | refresh | Verify current against post-RL-pipeline-merge state (memory step split, GRPO config, custom loss) |

## README freshness audit

A pre-rewrite audit step produces `tools/readme-assets/audit.md` (gitignored) with a discrepancy list driving both the main README rewrite and each subdir doc. Audit checks include but are not limited to:

- **Counts**: per-suite env counts (currently 71/63/57/50/43/16 = 300), per-suite action counts.
- **Install**: package name in `pyproject.toml`, extras names (`[eval]`, `[rl]`, `[all]`), `uv` syntax current.
- **APIs**: `make_env(...)` signature, `load_environment(task_id, num_episodes, n_frames, max_output_tokens, use_memory, memory_update_max_tokens)` matches current source.
- **Memory mode**: README claims "stored as one trajectory step"; recent commit `1c24d37` says it's split into two — confirm against current code.
- **CLI scripts**: every script path mentioned (`eval/run_debug.sh`, `eval/run_full.sh`, `scripts/rl/launch_all.sh`, `scripts/demo_all_envs.py`, `scripts/replay_trajectory.py`, `scripts/record_random_gifs.py`) exists and documented flags match.
- **Other paths**: `configs/rl/qwen35-4b-glyphbench/README.md`, `src/glyphbench/rl/README.md`, `eval/README.md`, `CONTRIBUTING.md`, `.env.cluster.template`, `eval/random_baseline.json` exist.
- **Project layout block**: tree matches `src/glyphbench/` reality.
- **External URLs**: leaderboard URL renders, GitHub URL is the intended public name.
- **Anonymization**: scan for `anon-paper-submission`, `"Anonymous"` author, etc. Flag, do not auto-rewrite.
- **Recent shipped features not yet in README**: e.g. craftax tutorial deliverable from phase γ.
- **Mentioned features no longer present**: anything still referenced but removed.

The audit is a single Markdown file with three sections: `## Stale facts to fix`, `## Judgment calls (need user input)`, `## Confirmed accurate`. The user reviews `## Judgment calls` before the rewrite proceeds.

## Pipelines

### Grid-only renderer (`tools/readme-assets/render_grid_gif.py`)

Pure Python, Pillow-based. Per env:
1. Instantiate via `make_env(env_id)`; reset with seed 42.
2. Run uniform-random rollout under env's natural `max_turns`.
3. For each step, extract `[Grid]` block from the observation string, plus header fields (turn, action, cumulative reward, env_id).
4. Render frame to PIL Image: dark background (RGB 15,15,15), 1-line header (white, monospace), then per-glyph colored grid using DejaVu Sans Mono at font_size 16.
5. After rollout terminates, save GIF (frame duration 200ms, loop forever).

CLI: `python render_grid_gif.py --output <dir> [--env <id>] [--suite <name>] [--overwrite]`. Defaults: render all 300, output to `tools/readme-assets/out/gifs/`.

### Full-TUI capture (asciinema → agg)

Per asset:
- `demo_all_envs.gif`: `asciinema rec --quiet --cols 140 --rows 40 -c "uv run scripts/demo_all_envs.py --env glyphbench/craftax-classic-v0 --delay 0.1" out.cast` → `agg --theme monokai --font-size 14 out.cast demo_all_envs.gif`.
- `gb_replay.gif`: **prerequisite** — `gb replay` consumes saved trajectory JSONLs (verifiers `results.jsonl` format), not random rollouts. Either reuse an existing saved trajectory under `cluster_manager/results/` (preferred — picks a real LLM rollout) or, if none cleanly fits the chosen env, spend Step 7 first generating a one-off via a quick `vf-eval` against a small local model targeted at `minigrid-doorkey-6x6-v0`. Then: hand-driven session under `asciinema rec --cols 140 --rows 40 out.cast`, executing `glyphbench replay <runs_dir> --pause --env glyphbench/minigrid-doorkey-6x6-v0`, with deliberate hotkey sequence `→ → s [pager-q] r [pager-q] m [pager-q] q`. Convert with `agg --theme monokai --font-size 14 out.cast gb_replay.gif`. Caption in README is a markdown line listing the keys.

### Hero composite (`tools/readme-assets/build_hero.sh`)

ffmpeg-concat the 12 chosen env GIFs with:
- 0.5s crossfade between each
- Bottom-strip caption per cell (e.g. `MiniGrid · DoorKey-6x6`) drawn with `drawtext` filter
- Final size ~900px wide, ~12s total loop

### Upload to HF

Reuse `scripts/upload_assets.py`:
- `upload_assets.py --src tools/readme-assets/out/gifs --dst gifs` for the 300
- `upload_assets.py --src tools/readme-assets/out/readme --dst readme` for the 3 README assets

## Execution order

```
Step 0.  Audit current README + intended subdir docs against codebase.
         Produce tools/readme-assets/audit.md with discrepancy list.
         Pause + show user the "Judgment calls" section; get OK before proceeding.

Step 1.  Install asciinema + agg locally; verify versions.

Step 2.  Build glyph→color palette + grid-only renderer.

Step 3.  VALIDATION GATE — render 3 sample envs only:
         craftax-classic-v0 (large grid), minigrid-empty-5x5-v0 (small),
         minihack-room-monster-15x15-v0 (info-dense). Show user; get explicit OK.

Step 4.  Render all 300 grid-only GIFs.

Step 5.  Build hero.gif (ffmpeg-concat 12 chosen envs + crossfades + captions).

Step 6.  Record demo_all_envs.gif (asciinema → agg, fully scripted).

Step 7.  Record gb_replay.gif. Prerequisite: locate a saved trajectory JSONL
         that targets `minigrid-doorkey-6x6-v0` under cluster_manager/results/;
         if none exists, generate a one-off via a small local vf-eval first.
         Then hand-drive asciinema with the hotkey sequence, then agg.

Step 8.  Upload all GIFs to HF (gifs/ × 300, readme/ × 3) via upload_assets.py.

Step 9.  Write the new main README + each new/refreshed subdir doc.
         Incorporate ALL audit fixes from Step 0.

Step 10. Public-release readiness check:
            - README renders correctly (gh markdown preview + branch view)
            - All HF asset URLs resolve (curl HEAD on each)
            - LICENSE present, CONTRIBUTING.md present
            - No leftover TODOs / scratch in any doc
            - Pause + show user the rendered branch view before commit

Step 11. Cleanup — rm -r tools/readme-assets/ (all temp tooling, audit.md, output dir).
         Commit. Done.
```

## What stays under user eye (not autonomous)

- **Step 0 (audit "Judgment calls")** — anonymization, GitHub URL, package rename: explicit user choice.
- **Step 3 (validation gate)** — explicit user OK on the 3 sample renders before proceeding to all 300.
- **Step 7 (gb_replay manual recording)** — interactive TTY can't be reliably scripted; user presses the hotkey sequence (or runs the prepared command and confirms the result).
- **Step 10 (final review)** — branch pushed, user eyeballs the rendered README on GitHub before commit.
- **Final commit** — only after Step 10 passes.

## Out of scope

- Repo-wide lint / type / test cleanup. ("just generate these aesthetics")
- GitHub Actions CI workflow. ("just generate these aesthetics")
- Modifying `record_random_gifs.py` / `replay_trajectory.py` / `upload_assets.py` (existing scripts kept as-is; new throwaway pipeline supersedes them for README purposes).
- Custom SVG/PNG banner above the hero GIF.
- Renaming the HF dataset repo away from `anon-paper-submission/glyphbench-assets`.
- Touching the leaderboard site (`docs/leaderboard/`) — though it auto-benefits from the better GIFs at the same HF URLs.
- Anything mentioned as a "Judgment call" in audit.md — those wait for user input.

## Risks

- **`agg` font fidelity**: agg ships its own font handling; some Unicode glyphs may render differently than the live terminal. Validation gate exposes this; we swap fonts (`--font-family` / preinstall a font) if needed.
- **Random rollout length variance**: some envs may produce 1-frame or 500-frame trajectories, producing degenerate GIFs. The 3-env gate doesn't cover this fully — we'll inspect a uniform sample after the all-300 render and re-roll any obviously degenerate ones with a different seed.
- **HF rate limit on bulk upload**: 300 small files = fine, but if it throttles we batch in groups.
- **README rendering on github.com vs locally**: GitHub markdown does its own thing with HTML alignment / `<details>` tags. Step 10 explicitly verifies on the branch URL, not just locally.
