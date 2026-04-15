#!/usr/bin/env bash
# scripts/serve_vllm.sh
# Convenience launcher for a vLLM OpenAI-compatible server.
#
# Usage:
#   scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct [port]
#
# The vLLM server runs in its own uv environment outside this project's venv
# because it has heavy CUDA dependencies. You must install vllm separately:
#   pip install vllm
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-8000}"

if ! command -v vllm >/dev/null 2>&1; then
    echo "error: 'vllm' command not found. Install vllm in a separate environment:" >&2
    echo "  uv venv --python 3.11 ~/venvs/vllm && source ~/venvs/vllm/bin/activate && uv pip install vllm" >&2
    exit 1
fi

echo "Starting vLLM server for model=${MODEL} on port=${PORT}"
exec vllm serve "${MODEL}" --port "${PORT}"
