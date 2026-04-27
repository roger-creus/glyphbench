# GlyphBench evaluation

GlyphBench exposes a verifiers environment with entry point
`glyphbench.load_environment`. Eval runs via the standard `vf-eval` CLI
against any OpenAI-compatible inference endpoint (we use vLLM).

## Quick start

```bash
# 1) start a vLLM server
uv run vllm serve Qwen/Qwen3.5-4B --port 8000

# 2) smoke test (1 env × 2 episodes)
bash eval/run_debug.sh

# 3) full eval (292 envs × N episodes)
bash eval/run_full.sh
```

## `load_environment` arguments

```python
load_environment(
    task_id: str | list[str] | None = None,  # single id, list, or None=all
    num_episodes: int = 5,                    # rollouts per env
    n_frames: int = 0,                        # history window (default 0 =
                                              # stateless per turn — every
                                              # observation must be readable
                                              # off the current grid alone)
    max_turns: int | None = None,             # None = use each env's own max
    max_output_tokens: int = 512,             # LLM budget per turn (advertised
                                              # to the model in the system
                                              # prompt — match your --max-tokens)
    seed: int = 42,
    use_memory: bool = False,                 # opt-in carried memory scaffold
    memory_update_max_tokens: int | None = None,
)
```

`task_id` (not `env_id`) — verifiers reserves `env_id` for the package
name passed via `vf.load_environment`; using `task_id` for the per-game
selector avoids the kwarg collision.

Pass as JSON to `-a` / `--env-args`:

```bash
vf-eval glyphbench \
  -m Qwen/Qwen3.5-4B \
  -b http://localhost:8000/v1 \
  -k OPENAI_API_KEY_LOCAL \
  -n 5 --max-tokens 8192 \
  -a '{"task_id":"glyphbench/atari-pong-v0","num_episodes":5,"n_frames":0,"max_output_tokens":8192}'
```

> **Tip.** Always pass `max_output_tokens` matching your `--max-tokens`. The
> system prompt advertises this value to the model so it can self-pace its
> reasoning; mismatched budgets cause premature self-truncation.

`use_memory=True` enables a two-generation environment turn: action selection
followed by a concise memory update conditioned on the action response and
environment feedback, including the HUD-stripped next-observation view. In
saved trajectories and RL training this remains one environment step; the
memory-update user prompt is context only, while the action assistant and
memory-update assistant tokens both receive the same task reward.
`memory_update_max_tokens` optionally changes only the second generation's token
limit.

## Results

Verifiers writes per-rollout JSON records and aggregate metrics under
`~/.prime/evals/…` by default. View with `prime eval tui` or replay with
`glyphbench replay <results-dir>`.

## Random baseline

`eval/random_baseline.json` is a zero-skill reference. Regenerate with:

```bash
uv run python eval/random_baseline.py --episodes 25 --output eval/random_baseline.json
```
