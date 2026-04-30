#!/usr/bin/env bash
# Multi-node smoke test launcher. Uses the full 3-node compute (1 trainer
# FSDP=8 + 16 vLLM workers across 2 inference nodes) on a SHORT run that
# exercises every code path: rollout generation, custom advantage with
# per-env Welford σ, sequence-mean loss, NCCL weight broadcast, checkpoint
# save, eval-during-training, W&B (offline) metric logging.
#
# Run from the trainer node (sf-node-1):
#   bash scripts/rl/launch_smoke_multinode.sh
#
# Tmux sessions: vllm-<host>, orch, trainer.

set -euo pipefail

# Use the production model (Qwen3.5-4B) so the smoke truly mirrors the
# eval profile — glyphbench's parser is tuned to Qwen3.5's thinking-token
# format, and using a different model would mask format-related bugs.
export CONFIG=configs/rl/qwen35-4b-glyphbench/smoke-multinode.toml
export MODEL=Qwen/Qwen3.5-4B
export MAX_MODEL_LEN=16384

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$REPO_ROOT/scripts/rl/launch_all.sh"
