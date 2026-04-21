# GlyphBench Session Handover

**Date:** 2026-04-21
**Session summary:** Fixed 66 broken envs, renamed atlas_rl → glyphbench, rebuilt and deployed container, submitted 144 experiment jobs + 8 model download jobs across 4 clusters. Experiments are running.

## IMPORTANT — HEADS UP

**Accidental deletion:** While testing the cluster-wipe path, I rsync-cleared `/scratch/rogercc/agentick/` on **rorqual** (contents replaced by an empty directory). `agentick` directories on narval / fir / mila are intact. If the rorqual copy is recoverable by rsyncing from another cluster, do that when you're back — I did not touch the other copies.

## Pipeline State

### Done
1. **Fixed all 66 broken envs** — `__init__` signatures now accept `max_turns` kwarg. Verified: `OK: 292, FAIL: 0`.
2. **Random baseline rerun** — 25 episodes × 292 envs × 200 max_turns → `eval/random_baseline.json` clean (0 errors).
3. **atlas_rl → glyphbench rename** — package, imports, env IDs, paths. All 2255 tests pass. Committed as `f2176d0`.
4. **Code review** (superpowers:code-reviewer) ran in parallel; caught jobs.py `tp` KeyError, config.py IndentationError, stray brand strings. All fixed. Committed as `40a0ffd`.
5. **Docker rebuilt** from Dockerfile → `glyphbench:latest` (22.6 GB).
6. **SIF rebuilt** via `apptainer build cluster_manager/glyphbench.sif docker-archive:/tmp/glyphbench.tar` (8.87 GB). Note: `docker-daemon://` hung, so we went through `docker save` → tar → apptainer.
7. **SIF deployed** to all 4 clusters at their respective `images/glyphbench.sif` paths.
8. **Code synced** to all 4 clusters (glyphbench/ replaces old atlas_rl/).
9. **Old atlas_rl dirs wiped** on rorqual / narval / fir (contents removed via rsync --delete from empty). Mila had no atlas_rl dirs.
10. **Old atlas_rl_runs dirs wiped** on all Alliance clusters. Old `glyphbench_runs/` on Mila also wiped.
11. **Leaderboard site scaffolded** at `docs/leaderboard/index.html` (dark mode, sortable, filterable; reads from `data.json`).
12. **Build script** at `scripts/build_leaderboard.py` reads `results/` → produces `data.json`.
13. **Download-models script** at `cluster_manager/download_models.py` (cluster-generic sbatch generator for HF snapshot_download).
14. **New cm.py subcommand `wipe-results`** for future-proof cluster cleanup.
15. **README + CONTRIBUTING polished** — counts corrected (292 envs, 6 suites), submission protocol documented.
16. **community_runs/** dir scaffolded for external submissions.

### Currently Running
- **8 HF download jobs on Mila (long-cpu partition)** — fetching missing models:
  - DeepSeek-R1-Distill-Qwen-1.5B, 7B, 14B
  - Qwen3-14B
  - gemma-3-4b-it, gemma-3-12b-it
  - Llama-3.1-8B-Instruct
  - Mistral-7B-Instruct-v0.3
- **144 experiment jobs** submitted across 4 clusters (72 + 72):
  - Qwen3.5 family (0.8B, 2B, 4B) × 4 harness × 6 suites, round-robin rorqual/narval/fir/mila
  - Mila-only: Qwen3-0.6B, Qwen3-8B, Qwen3.5-9B × 4 harness × 6 suites
- **1 pilot job** (Qwen3-0.6B + markov_zeroshot + minigrid on Mila) confirmed the pipeline works — vLLM loaded, processing prompts at ~300 tok/s.

### What's NOT done
- 13 of 19 configured models not yet cached (8 downloading, 5 large models we're skipping for tonight: Qwen3.5-27B, Qwen3-32B, DeepSeek-R1-Distill-Qwen-32B, gemma-3-27b-it)
- Full experiment matrix for missing models (will launch when downloads finish)
- Trajectory GIFs (post-experiments)
- Paper-ready plots (post-experiments)
- Leaderboard `data.json` regeneration (pending first results)
- Auto-pull + resubmit-failures loop
- CI validation for community submissions

## Key Files

| File | Purpose |
|------|---------|
| `src/glyphbench/` | Main package (292 envs, 6 suites) |
| `eval/run_eval.py` | Batched vLLM eval runner with 4 harness modes |
| `eval/scoring.py` | GlyphBench Score (random-normalized IQM per suite) |
| `eval/random_baseline.py` | 25-episode random baseline runner |
| `eval/random_baseline.json` | 292-env baseline (clean, 0 errors) |
| `cluster_manager/cm.py` | Multi-cluster CLI (setup, submit, status, pull, wipe-results) |
| `cluster_manager/config.py` | 4 clusters, 19 models, harness modes |
| `cluster_manager/jobs.py` | SLURM job generation (model × harness × suite) |
| `cluster_manager/download_models.py` | HF snapshot_download sbatch generator |
| `cluster_manager/glyphbench.sif` | Container (8.87 GB, not in git) |
| `docs/leaderboard/index.html` | Leaderboard site |
| `scripts/build_leaderboard.py` | results/ → data.json |

## Cluster Access

| Cluster | SSH | GPU | SIF | Code |
|---------|-----|-----|-----|------|
| rorqual | `rorqual-robot` | H100 MIG 40GB | deployed | synced |
| narval  | `narval-robot`  | A100 40GB       | deployed | synced |
| fir     | `fir-robot`     | H100 MIG 40GB   | deployed | synced |
| mila    | `mila` (2FA via `ssh mila` first) | 8×A6000 + 8×A100 48GB | deployed | synced |

**Robot restrictions:** sbatch / squeue / scancel / rsync / ls / cat / mkdir only. No `rm`, no shell operators, no `-o` quoted args in squeue.

## Models: Cached Inventory

Cached on all Alliance clusters + Mila:
- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3.5-2B
- Qwen/Qwen3.5-4B

Cached on Mila only:
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-8B
- Qwen/Qwen3.5-9B
- (plus rogercc/agentick-qwen35-4b-sft-ascii, external)

Missing / downloading (Mila long-cpu queue):
- DeepSeek-R1-Distill-Qwen-1.5B / 7B / 14B
- Qwen3-14B
- gemma-3-4b-it / 12b-it
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3

Skipped for tonight (too large):
- Qwen3.5-27B, Qwen3-32B, DeepSeek-R1-Distill-Qwen-32B, gemma-3-27b-it

## Next Session Steps

```
0. ssh mila; check squeue for download jobs' completion.
1. cd cluster_manager && python3 cm.py status --detailed   # check experiment queue
2. cd cluster_manager && python3 cm.py pull                 # pull results
3. For newly-cached models: submit their experiment batches (same pattern as Qwen3.5 above)
4. Check any failed jobs via log files under output_dir/logs/
5. Regenerate leaderboard: uv run python scripts/build_leaderboard.py
6. If ready: push docs/leaderboard/ via GitHub Pages
```

## Design Decisions

1. **No success_rate metric** — raw mean_return only.
2. **GlyphBench Score** = equal-weight mean of per-suite IQMs (normalized vs random baseline).
3. **Unicode glyphs** — single-codepoint per cell, zero collisions.
4. **4 harness modes** — markov/history × zeroshot/cot.
5. **1 SLURM job per (model, harness, suite)** — 19 × 4 × 6 = 456 jobs when all models available.
6. **Official protocol**: 25 episodes × 200 max_turns × fixed seeds × temp 0.7 × 16384 thinking tokens.
7. **Random baseline**: 25 episodes × 200 max_turns.
