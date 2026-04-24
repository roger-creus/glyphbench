#!/usr/bin/env bash
# Full eval: all 293 envs × 10 episodes × Qwen3-0.6B.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}

uv run vf-eval glyphbench \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -k "${API_KEY_VAR:-VLLM_API_KEY}" \
  -n 10 -t 512 \
  -a '{"num_episodes": 10, "n_frames": 4}'
