#!/usr/bin/env bash
# Full eval: all 300 envs × $EPISODES (default 10) × $MODEL.
#
# Prereq: a vLLM server at $VLLM_BASE_URL serving the target model. e.g.:
#   uv run vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 24576
#
# 24576 = 16384 input cap + 8192 action output budget. Memory turns send
# max_tokens=4096, comfortably below the same envelope.
#
# After it finishes, view results with: prime eval tui

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
API_KEY_VAR=${API_KEY_VAR:-OPENAI_API_KEY_LOCAL}
EPISODES=${EPISODES:-5}
N_FRAMES=${N_FRAMES:-0}
MAX_TOKENS=${MAX_TOKENS:-8192}

export "$API_KEY_VAR=${!API_KEY_VAR:-EMPTY}"

prime eval run glyphbench \
  --provider vllm \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -k "$API_KEY_VAR" \
  -n "$EPISODES" \
  --max-tokens "$MAX_TOKENS" \
  -a "{\"num_episodes\": $EPISODES, \"n_frames\": $N_FRAMES, \"max_output_tokens\": $MAX_TOKENS}"
