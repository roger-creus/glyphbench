#!/usr/bin/env bash
# Full eval over the active subset of GlyphBench.
#
# By default this excludes the archival ``atari`` (long-horizon originals)
# and ``craftaxfull`` (open-ended Crafter) suites. To include them, run
# ``eval/run_archival.sh`` or pass ``EXCLUDE_SUITES='[]'``.
#
# Prereq: a vLLM server at $VLLM_BASE_URL serving the target model. e.g.:
#   uv run vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 32768
#
# Budget arithmetic (memory mode is the default):
#   action call:  prompt (sys+obs) + 8192 action_output_tokens   <= max-model-len
#   memory call:  prompt (sys+obs+action_tag+lean_user) + 4096   <= max-model-len
#                 where lean_user re-injects the action reasoning (<=8192 tok)
#                 plus the next-obs grid (~1500 tok) and the [Memory Update]
#                 instruction.
# 32768 covers the worst case: ~8K prompt + 8K reasoning re-injection +
# 1.5K next-obs + 0.5K wrappers + 4K memory output budget = ~22K, with
# margin for craftax-sized grids and longer system prompts.
#
# After it finishes, view results with: prime eval tui

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
API_KEY_VAR=${API_KEY_VAR:-OPENAI_API_KEY_LOCAL}
EPISODES=${EPISODES:-5}
N_FRAMES=${N_FRAMES:-0}
MAX_TOKENS=${MAX_TOKENS:-8192}
EXCLUDE_SUITES=${EXCLUDE_SUITES:-'["atari","craftaxfull"]'}

export "$API_KEY_VAR=${!API_KEY_VAR:-EMPTY}"

prime eval run glyphbench \
  --provider vllm \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -k "$API_KEY_VAR" \
  -n "$EPISODES" \
  --max-tokens "$MAX_TOKENS" \
  -a "{\"num_episodes\": $EPISODES, \"n_frames\": $N_FRAMES, \"max_output_tokens\": $MAX_TOKENS, \"exclude_suites\": $EXCLUDE_SUITES}"
