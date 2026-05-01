# eval/

Verifiers-driven evaluation of every GlyphBench env against any
OpenAI-compatible endpoint. Scoring is raw episodic return per (env, model)
— no benchmark-wide normalised aggregate.

## Quick start

```bash
# Wire-check: 1 env, 2 episodes
bash eval/run_debug.sh

# Full sweep: all 343 envs, configurable via env vars
bash eval/run_full.sh
```

Both scripts assume an OpenAI-compatible server is reachable at
`http://localhost:8000/v1`. Easiest local server:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 24576
```

## CLI flags

Both scripts are driven entirely by environment variables. Set them before
calling the script or prefix inline.

### Shared variables (both scripts)

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | HF model ID passed to `--model`. |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL. |
| `API_KEY_VAR` | `OPENAI_API_KEY_LOCAL` | Name of the env var holding the API key (vLLM accepts any non-empty value). |
| `N_FRAMES` | `0` | Frame-stack history window. `0` = stateless per turn. |
| `MAX_TOKENS` | `8192` | Per-turn output token budget (`--max-tokens`). Also forwarded as `max_output_tokens` to the env. |

### `run_debug.sh` — additional variables

| Variable | Default | Description |
|---|---|---|
| `NUM_EPISODES` | `2` | Episodes per env (passed as `-n` and into `num_episodes`). |
| `ROLLOUTS_PER_EXAMPLE` | `1` | Repeated rollouts per (env, seed) row. |
| `TASK_IDS` | `["glyphbench/minigrid-empty-5x5-v0"]` | JSON array of task IDs to evaluate (single-env smoke test). |
| `TEMPERATURE` | `1.0` | Sampling temperature. |
| `SAMPLING_ARGS` | `{"top_p":0.95,...}` | JSON blob forwarded to `--sampling-args` (Qwen3.5 thinking profile). |
| `AUTO_START_VLLM` | `0` | Set to `1` to let the script launch vLLM in the background and wait up to 10 min. |
| `VLLM_LOG` | `/tmp/glyphbench-vllm.log` | Log path when `AUTO_START_VLLM=1`. |

### `run_full.sh` — additional variables

| Variable | Default | Description |
|---|---|---|
| `EPISODES` | `5` | Episodes per env (`-n` and `num_episodes`). |

**Examples:**

```bash
# Debug: different model, 4 episodes, custom task
MODEL=Qwen/Qwen3-1.7B NUM_EPISODES=4 TASK_IDS='["glyphbench/atari-pong-v0"]' bash eval/run_debug.sh

# Debug: let the script start vLLM automatically
AUTO_START_VLLM=1 bash eval/run_debug.sh

# Full: 10 episodes, no frame stack
EPISODES=10 N_FRAMES=0 bash eval/run_full.sh

# Full: remote server
VLLM_BASE_URL=http://gpu-host:8000/v1 bash eval/run_full.sh
```

Results are written to `~/.prime/evals/`. View with `prime eval tui`.

## Running a single env from Python

```python
import glyphbench
vf_env = glyphbench.load_environment(
    task_id="glyphbench/minigrid-empty-5x5-v0",
    num_episodes=5,
    n_frames=0,
    max_output_tokens=8192,  # match your --max-tokens
    use_memory=False,
)
```

The returned `verifiers.Environment` plugs into `prime eval run` or any
RL trainer that consumes verifiers envs.

Full signature:

```python
load_environment(
    task_id: str | list[str] | None = None,  # single id, list, or None = all envs
    num_episodes: int = 5,                    # rollouts per env
    n_frames: int = 0,                        # history window (0 = stateless)
    max_turns: int | None = None,             # None = use each env's own budget
    max_output_tokens: int = 512,             # LLM token budget per turn
    seed: int = 42,
    use_memory: bool = False,                 # opt-in carried memory scaffold
    memory_update_max_tokens: int | None = None,  # token cap for memory update only
)
```

`task_id` (not `env_id`) — verifiers reserves `env_id` for the package
name; `task_id` is the per-game selector and avoids the kwarg collision.

Always pass `max_output_tokens` matching your server's `--max-tokens`. The
system prompt advertises this value so the model can self-pace its reasoning;
mismatched values cause premature self-truncation.

`use_memory=True` enables a two-generation turn: action selection followed by
a concise memory update. `memory_update_max_tokens` caps only the second
generation's output. See `docs/OBSERVATION_FORMAT.md` for full memory-mode
behaviour.

## Filtering envs (suites and tasks)

Both `prime eval run glyphbench` and `random_baseline.py` accept include/exclude
filters. `glyphbench.load_environment` exposes them as kwargs; eval scripts pass
them through the `-a '{...}'` JSON or as CLI args.

| kwarg | type | behavior |
|---|---|---|
| `include_suites` | `list[str]` | Whitelist by first-hyphen-segment of env_id (e.g. `["atari", "minigrid"]`). |
| `exclude_suites` | `list[str]` | Blacklist by suite. Always wins over includes. |
| `include_tasks` | `list[str]` | Exact env IDs or fnmatch patterns (e.g. `["glyphbench/atari-*-v0"]`). |
| `exclude_tasks` | `list[str]` | Same shape; always wins over includes. |

`__dummy` envs are excluded by default. To include them, name them explicitly
in `include_tasks`.

**Default eval exclusions.** `eval/run_full.sh` defaults to
`exclude_suites=["atari", "craftaxfull"]` because those two suites are the
long-horizon archival/open-ended games. To eval them, use `eval/run_archival.sh`
or pass `EXCLUDE_SUITES='[]'`.

**Default training exclusions.** Training configs (prime-rl TOML) should set
`exclude_suites=["atari", "craftaxfull"]` for the same reason.

## Random-agent baseline

A reproducible zero-skill reference is shipped at `eval/random_baseline.json`.
It contains the expected return per env under uniform-random action selection
at each env's natural `max_turns` budget.

Schema per entry:

```json
{
  "glyphbench/atari-alien-v0": {
    "env_id": "glyphbench/atari-alien-v0",
    "n_episodes": 5,
    "mean_return": 82.0,
    "std_return": 56.36,
    "min_return": 10.0,
    "max_return": 170.0,
    "median_return": 60.0,
    "mean_length": 110.8,
    "per_episode_returns": [60.0, 120.0, 170.0, 10.0, 50.0]
  }
}
```

Regenerate with:

```bash
uv run python eval/random_baseline.py
# options:
#   --episodes 25            episodes per env (default 25)
#   --include-suite NAME     restrict to a suite (repeatable)
#   --exclude-suite NAME     skip a suite (repeatable; default: atari, craftaxfull)
#   --include-task ID|GLOB   restrict by exact id or fnmatch (repeatable)
#   --exclude-task ID|GLOB   skip by exact id or fnmatch (repeatable)
#   --include-all            disable default suite exclusions
#   --max-turns N            override env-native budget (default: None = use env's own)
#   --output eval/random_baseline.json
```

The shipped file uses `--episodes 25` with each env's natural max_turns.
Passing `--max-turns` distorts the reference for envs whose difficulty
depends on their step budget, so the default leaves it unset.

## Scoring

GlyphBench reports **raw episodic return per (env, model)**. There is no
benchmark-wide normalised score: per-task per-model means are published
raw and downstream analyses choose their own aggregation. The leaderboard
site under `docs/leaderboard/` aggregates submitted results; see the
project root README for how to submit.

## Failure modes & metrics

Every rollout reports a fixed set of observability metrics. See
[docs/llm-agent-failure-modes.md](../docs/llm-agent-failure-modes.md) for
definitions and diagnostic guidance.

Common signals:

- `forfeit_rate` — agent failed to emit `<action>NAME</action>`.
- `action_completion_truncation_rate` — reasoning hit the 8192-token cap.
- `memory_completion_truncation_rate` — memory write hit the 4096-token cap.
- `memory_parse_failure_rate` — memory wasn't wrapped in `<memory>` tags.
- `episode_terminated_rate` / `episode_truncated_max_turns_rate` — natural end vs ran-out-of-clock.

vLLM server requirement: `--max-model-len 24576` (= 16384 input cap + 8192
action output budget). Memory turns send `max_tokens=4096`.

## Files

| File | Description |
|---|---|
| `run_debug.sh` | Smoke eval: 1 env, 2 episodes, optional auto-start vLLM. |
| `run_full.sh` | Full sweep over the active subset (excludes `atari`, `craftaxfull` by default). |
| `run_archival.sh` | Archival sweep: includes only the long-horizon `atari` + `craftaxfull` suites. |
| `random_baseline.json` | Pre-computed zero-skill reference: mean/std/min/max return per env under uniform-random action selection. |
| `random_baseline.py` | Script to regenerate `random_baseline.json`. See "Filtering envs" for filter flags. |
| `plot_results.ipynb` | Notebook for per-suite result visualisation. |
| `eval/figures/` | Auto-regenerated per-suite plots (gitignored). |
