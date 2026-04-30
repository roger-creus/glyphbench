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

# Smaller / faster model to validate plumbing on first pass. Once this
# smoke passes, swap to MODEL=Qwen/Qwen3.5-4B + MAX_MODEL_LEN=16384 +
# CONFIG=configs/rl/qwen35-4b-glyphbench/rl.toml for production.
export CONFIG=configs/rl/qwen35-4b-glyphbench/smoke-multinode.toml
export MODEL=Qwen/Qwen3-4B-Instruct-2507
export MAX_MODEL_LEN=8192

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$REPO_ROOT/scripts/rl/launch_all.sh"
