#!/usr/bin/env bash
# Smoke eval: 2 envs × 2 episodes × Qwen3-0.6B against a local vLLM server.
# Prereq: a vLLM server at $VLLM_BASE_URL (default http://localhost:8000/v1)
# serving Qwen/Qwen3-0.6B. Start with:
#   uv run vllm serve Qwen/Qwen3-0.6B --port 8000
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}

uv run vf-eval glyphbench \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -k "${API_KEY_VAR:-VLLM_API_KEY}" \
  -n 2 -t 512 \
  -a '{"task_id": ["glyphbench/__dummy-v0", "glyphbench/minigrid-empty-5x5-v0"], "num_episodes": 2, "n_frames": 4}'
