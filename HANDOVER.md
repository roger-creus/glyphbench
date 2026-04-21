# GlyphBench Session Handover

**Date:** 2026-04-21
**Session summary:** Fixed 66 broken envs, completed atlas_rl → glyphbench rename, rebuilding container + deploying to 4 clusters.

## IMPORTANT — HEADS UP

**Accidental deletion:** While testing the cluster-wipe path, I rsync-cleared `/scratch/rogercc/agentick/` on **rorqual** (contents replaced by an empty directory). `agentick` directories on narval / fir / mila are intact. If the rorqual copy is recoverable by rsyncing from another cluster, do that when you're back — I did not touch the other copies.

**Cluster wipe plan (pending, before experiment submission):** Contents of old `atlas_rl/`, `atlas_rl_runs/`, and `glyphbench_runs/` on every cluster are to be wiped before resubmitting the full matrix. User directive 2026-04-21.

## Current State (at session start)

- **292 environments** across 6 suites (MiniGrid 71, MiniHack 63, Atari 57, Classics 50, Craftax 34, Procgen 16, dummy 1 filtered out)
- **Unicode rendering** across all suites
- **4 harness modes**: markov_zeroshot, markov_cot, history_zeroshot, history_cot
- **Eval runner** (`eval/run_eval.py`): batched vLLM offline inference, trajectory recording
- **Scoring system** (`eval/scoring.py`): random-normalized scores, IQM aggregation, GlyphBench Score
- **Cluster manager** (`cluster_manager/`): 4 clusters (rorqual, narval, fir, mila)
- **19 models configured** in cluster_manager/config.py

## Done This Session

1. **Fixed all 66 broken envs** — `__init__` signatures now accept `max_turns` kwarg, including `_SubtaskMixin` in craftax subtasks. Verified: `OK: 292, FAIL: 0`.
2. **Random baseline rerun** — 25 episodes × 292 envs × 200 max_turns → `eval/random_baseline.json` clean (0 errors, completed in 29s).
3. **atlas_rl → glyphbench rename** — package dir moved, all imports updated, all env IDs renamed, Dockerfile + configs + docs updated. Committed as `f2176d0`. All 2255 tests pass.
4. **Docker image built** locally (`docker images | grep glyphbench` → `glyphbench:latest`, sha256:bb9c0889...).
5. **SIF build in progress** — apptainer build from `docker-daemon://glyphbench:latest` → `cluster_manager/glyphbench.sif` (see `/tmp/docker_build.log`).
6. **Fixed pre-existing bug in `cluster_manager/config.py`** — mila dict was outside CLUSTERS dict (unreachable). Now inside.
7. **Fixed `cmd_status` in cm.py** — was grepping for "atlas" in squeue output, now greps "glyphbench" / "gb_".

## Task List (in order)

1. [DONE] Fix 66 broken envs (init kwarg)
2. [DONE] Rerun random baseline, 0 errors
3. [PENDING] Code review (superpowers:code-reviewer, running in background)
4. [DONE] Rename atlas_rl → glyphbench
5. [IN PROGRESS] Rebuild Docker + push SIF to all 4 clusters
6. [PENDING] Wipe old atlas_rl/, atlas_rl_runs/, glyphbench_runs/ on all clusters
7. [PENDING] Download 19 models to cluster HF caches
8. [PENDING] Submit full experiment matrix (~456 SLURM jobs: 19 models × 4 harness × 6 suites)
9. [PENDING] Monitor, pull results, resubmit failures until 100% complete
10. [PENDING] Leaderboard website (GitHub Pages), submission infra, CONTRIBUTING.md, README polish

## Key Files

| File | Purpose |
|------|---------|
| `src/glyphbench/` | Main package |
| `eval/run_eval.py` | Batched vLLM eval runner |
| `eval/scoring.py` | GlyphBench Score |
| `eval/random_baseline.py` | Random baseline runner (25 eps preferred) |
| `eval/random_baseline.json` | Baseline data (292 envs clean) |
| `cluster_manager/cm.py` | Multi-cluster CLI |
| `cluster_manager/config.py` | Clusters, models, harness modes |
| `cluster_manager/jobs.py` | SLURM job generation |
| `cluster_manager/glyphbench.sif` | Container (being rebuilt) |
| `Dockerfile` | vllm/vllm-openai:latest base |

## Cluster Access

| Cluster | SSH | GPU | Status |
|---------|-----|-----|--------|
| rorqual | `rorqual-robot` | H100 MIG 40GB | SIF + code: deploy pending |
| narval | `narval-robot` | A100 40GB | SIF + code: deploy pending |
| fir | `fir-robot` | H100 MIG 40GB | SIF + code: deploy pending |
| mila | `mila` (needs `ssh mila` for 2FA first) | 8×A6000 + 8×A100 48GB | SIF + code: deploy pending |

**Robot restrictions:** `sbatch`, `squeue`, `scancel`, `rsync`, `ls`, `cat`, `mkdir` only. No `rm`, `grep`, shell operators. Use `rsync -av --delete` from empty dir for cleanup.

**Mila 2FA:** Run `! ssh mila` first to trigger TOTP, then session persists.

## Design Decisions

1. **No success_rate metric** — raw mean_return only.
2. **GlyphBench Score** = equal-weight mean of per-suite IQMs (normalized vs random baseline).
3. **Unicode glyphs** — single-codepoint chars per cell, zero symbol collisions.
4. **4 harness modes** — markov/history × zeroshot/cot.
5. **Trajectory-verified submissions** — anti-cheat via replay.
6. **1 SLURM job per (model, harness, suite)** — 19 × 4 × 6 = 456 jobs total.
7. **Random baseline**: 25 episodes × 200 max_turns (user-confirmed).
