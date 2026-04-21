# GlyphBench Session Handover

**Date:** 2026-04-21
**Session summary:** Built the entire GlyphBench benchmark from scratch -- 293 environments, eval infrastructure, cluster deployment, scoring system.

## Current State

### What's done
- **293 environments** across 7 suites (MiniGrid 71, MiniHack 63, Atari 57, Classics 50, Craftax 35, Procgen 16, dummy 1)
- **Unicode rendering** across all suites (arrows, blocks, distinct glyphs per entity)
- **4 harness modes**: markov_zeroshot, markov_cot, history_zeroshot, history_cot
- **Eval runner** (`eval/run_eval.py`): batched vLLM offline inference, trajectory recording, 16384 token thinking budget
- **Scoring system** (`eval/scoring.py`): random-normalized scores, IQM aggregation, GlyphBench Score, pairwise win rates
- **Random baseline** (`eval/random_baseline.json`): 226/292 envs baselined (66 have bugs, see below)
- **Cluster manager** (`cluster_manager/`): supports 4 clusters (rorqual, narval, fir, mila)
- **Container** (`cluster_manager/glyphbench.sif`): built from vllm/vllm-openai:latest, 8.3GB
- **SIF deployed** to: narval, rorqual, fir, mila
- **Code synced** to: all 4 clusters
- **Paper plotting notebook** (`eval/plot_results.ipynb`): 7 figures + scoring integration
- **19 models configured** in cluster_manager/config.py
- **Memory files** updated for fresh session context

### Known bugs (66 envs failing in random baseline)
1. **Craftax sub-tasks (~52 errors)**: `__init__` signature mismatch when `gym.make()` passes `max_turns` kwarg. The subtask classes likely need to accept `**kwargs` or explicit `max_turns` parameter.
2. **Classics (~14 errors)**: artillery, flood_fill, guard_evasion, and others have `__init__` kwarg issues. Same root cause -- `gym.make(env_id, max_turns=200)` passes `max_turns` but the class `__init__` doesn't accept it.
3. **Fix pattern**: ensure all env `__init__` methods accept `max_turns` and pass it to `super().__init__(max_turns=max_turns)`.

### What's NOT done (task list)

**Critical path (sequential):**
1. **#8 Code review + correctness audit** -- fix the 66 broken envs, review all games for correct rewards/prompts/action spaces, simplify code
2. **#9 Rename glyphbench -> glyphbench** -- package dir, ALL imports, ALL env IDs, tests, scripts. ~1000 files. Use sed.
3. **#10 Rebuild container + push** -- after rename, rebuild Docker->SIF, push to all 4 clusters
4. **#11 Download 19 models** -- to cluster HF caches. Small models: download locally + rsync. Large models: download on Mila directly.
5. **#12 Submit full experiment matrix** -- ~500+ SLURM jobs. Monitor, pull, resubmit failures.
6. **#13 Generate trajectory GIFs** -- after experiments
7. **#17 Paper-ready plots** -- after experiments

**Parallel work:**
8. **#14 Leaderboard website** -- GitHub Pages, beautiful minimalistic dynamic page
9. **#15 Result submission system** -- trajectory-verified submissions, CI validation, anti-cheat
10. **#16 README + module READMEs** -- curated, elegant, benchmark-focused
11. **#18 Community contribution infra** -- game template, CONTRIBUTING.md, CI auto-tests

## Key Files

| File | Purpose |
|------|---------|
| `src/glyphbench/` | Main package (PENDING RENAME to glyphbench/) |
| `eval/run_eval.py` | Batched vLLM eval runner with 4 harness modes |
| `eval/scoring.py` | GlyphBench Score: normalization + IQM |
| `eval/random_baseline.py` | Random baseline runner |
| `eval/random_baseline.json` | Baseline results (226 working, 66 broken) |
| `eval/plot_results.ipynb` | 7-figure paper plotting notebook |
| `cluster_manager/cm.py` | Multi-cluster experiment manager CLI |
| `cluster_manager/config.py` | Clusters (4), models (19), harness modes, eval params |
| `cluster_manager/jobs.py` | SLURM job generation (1 job per model x harness x suite) |
| `scripts/demo_all_envs.py` | Interactive demo with Unicode+color |
| `scripts/replay_trajectory.py` | Trajectory replay + GIF export |
| `Dockerfile` | Container definition (FROM vllm/vllm-openai:latest) |

## Cluster Access

| Cluster | SSH | GPU | Status |
|---------|-----|-----|--------|
| rorqual | `rorqual-robot` (auto-auth) | H100 MIG 40GB | SIF + code deployed |
| narval | `narval-robot` (auto-auth) | A100 40GB | SIF + code deployed |
| fir | `fir-robot` (auto-auth) | H100 MIG 40GB | SIF + code deployed |
| mila | `mila` (needs `! ssh mila` for 2FA first) | 8xA6000 + 8xA100 48GB | SIF + code deployed |

## Design Decisions

1. **No success_rate metric** -- raw mean_return only. Success is env-specific.
2. **GlyphBench Score** = equal-weight mean of per-suite IQMs (normalized vs random baseline)
3. **Unicode glyphs** -- `→↓←↑` for player dirs, `█` walls, `★` goals, etc. Zero symbol collisions.
4. **4 harness modes** -- markov/history x zeroshot/cot. Core experimental design.
5. **Trajectory-verified submissions** -- anti-cheat via replay verification of full .jsonl trajectories.
6. **1 SLURM job per (model, harness, suite)** -- smart splitting, ~500 total jobs.

## First Thing To Do In New Session

```
Read memory/ files, then:
1. Fix the 66 broken envs (init kwarg issue)
2. Run full random baseline again (all 292 should pass)
3. Then continue down the task list
```
