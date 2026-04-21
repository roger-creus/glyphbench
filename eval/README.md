# GlyphBench Evaluation Pipeline

Batched LLM evaluation using vLLM offline inference. No server needed -- loads the model directly into GPU memory and runs batched inference across all active environments at each timestep.

## Setup

```bash
# Install with eval dependencies
uv add "glyphbench[eval]"

# Or from source
uv sync --extra eval
```

## Running Evaluations

```bash
# Full evaluation (all 210 envs, 25 episodes each)
uv run python eval/run_eval.py --config eval/configs/qwen35_4b.yaml

# Quick test on specific envs
uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-4B \
    --episodes 5 \
    --envs glyphbench/minigrid-empty-5x5-v0 glyphbench/atari-pong-v0

# Filter by suite
uv run python eval/run_eval.py --model Qwen/Qwen3.5-4B --suites minigrid atari

# Custom settings
uv run python eval/run_eval.py \
    --model Qwen/Qwen3.5-4B \
    --episodes 25 \
    --max-turns 200 \
    --batch-size 25 \
    --temperature 0.7 \
    --output results/
```

## Results Structure

```
results/
  Qwen_Qwen3.5-4B/
    results.json          # Aggregate results (all envs)
    per_env/
      glyphbench__minigrid-empty-5x5-v0.json
      glyphbench__atari-pong-v0.json
      ...
```

Each per-env JSON contains:
- `mean_return`, `std_return`, `min_return`, `max_return`
- `success_rate`
- `mean_length`
- `mean_parse_failures`
- Token usage and latency stats

## Plotting Results

```bash
jupyter notebook eval/plot_results.ipynb
```

## Configuration

See `eval/configs/qwen35_4b.yaml` for all available options:
- `model`: HuggingFace model ID
- `episodes`: Number of episodes per environment
- `max_turns`: Maximum steps per episode
- `batch_size`: How many episodes to batch per vLLM call
- `temperature`: Sampling temperature
- `max_new_tokens`: Max output tokens per LLM call
- `max_model_len`: vLLM max context length
- `gpu_memory_utilization`: Fraction of GPU memory for vLLM
- `dtype`: Model dtype (bfloat16, float16, auto)
- `enforce_eager`: Disable CUDA graphs (required for MIG GPU slices, default: true)
