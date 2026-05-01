#!/usr/bin/env bash
# Archival eval: runs the full-horizon atari originals + craftaxfull. Slow.
#
# Prereq: a vLLM server at $VLLM_BASE_URL serving the target model.
#
# Use this only when you specifically want to measure the open-ended /
# long-horizon performance ceiling. Default eval excludes these suites
# (see eval/run_full.sh).

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
  -a "{\"num_episodes\": $EPISODES, \"n_frames\": $N_FRAMES, \"max_output_tokens\": $MAX_TOKENS, \"include_suites\": [\"atari\",\"craftaxfull\"]}"
