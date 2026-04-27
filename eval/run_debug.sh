#!/usr/bin/env bash
# Smoke eval via `prime eval run` against a local vLLM server.
#
# This script will:
#   1. Pre-check vLLM is reachable at $VLLM_BASE_URL.
#   2. If not, optionally start one in the background (set AUTO_START_VLLM=1).
#   3. Run a 2-env × 2-episode smoke eval through `prime eval run`.
#
# Usage:
#   bash eval/run_debug.sh                           # vLLM must already be up
#   AUTO_START_VLLM=1 bash eval/run_debug.sh         # start vLLM if needed (~60s)
#   MODEL=Qwen/Qwen3-1.7B bash eval/run_debug.sh
#   VLLM_BASE_URL=http://other-host:8000/v1 bash eval/run_debug.sh
#
# After it finishes, view results with: prime eval tui

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project venv so `prime`, `vllm`, and `glyphbench` are on PATH/PYTHONPATH.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$REPO_ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Pull HF_TOKEN etc. from .env if present (for vLLM model downloads).
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
API_KEY_VAR=${API_KEY_VAR:-OPENAI_API_KEY_LOCAL}
AUTO_START_VLLM=${AUTO_START_VLLM:-0}
VLLM_LOG=${VLLM_LOG:-/tmp/glyphbench-vllm.log}

# Eval knobs (override via env)
N_FRAMES=${N_FRAMES:-0}                         # frame-stack history window (0 = stateless per turn)
NUM_EPISODES=${NUM_EPISODES:-2}                 # distinct (env, seed) rows per env; also -n
ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-1} # repeats per row
MAX_TOKENS=${MAX_TOKENS:-8192}                  # per-turn output budget (Qwen3.5 thinking floor)
TASK_IDS=${TASK_IDS:-'["glyphbench/minigrid-empty-5x5-v0"]'}

# Sampling — Qwen3.5 thinking profile (verified vs HF model cards 2026-04-25).
# Output budget needs ≥4K with thinking on; 8K is the recommended floor.
TEMPERATURE=${TEMPERATURE:-1.0}
SAMPLING_ARGS=${SAMPLING_ARGS:-'{"top_p":0.95,"presence_penalty":1.5,"extra_body":{"top_k":20,"min_p":0.0,"chat_template_kwargs":{"enable_thinking":true}}}'}

# vLLM doesn't validate the key — set a placeholder if not already set.
export "$API_KEY_VAR=${!API_KEY_VAR:-EMPTY}"

# ---------------------------------------------------------------------------
# vLLM reachability check
# ---------------------------------------------------------------------------

is_vllm_up() {
    curl -fs --max-time 3 "${BASE_URL%/v1}/v1/models" > /dev/null 2>&1
}

start_vllm_in_bg() {
    echo ">>> Starting vLLM: $MODEL on $BASE_URL  (logging to $VLLM_LOG)"
    local port
    port="$(echo "$BASE_URL" | sed -E 's|.*://[^:/]+:?([0-9]*)/?.*|\1|')"
    port="${port:-8000}"
    nohup vllm serve "$MODEL" --port "$port" --max-model-len 8192 \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "    vllm pid=$VLLM_PID"
    echo ">>> Waiting for vLLM to become ready (up to 10 min)..."
    local deadline=$((SECONDS + 600))
    while [ $SECONDS -lt $deadline ]; do
        if is_vllm_up; then
            echo ">>> vLLM ready"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "!!! vLLM process died. Last 30 lines of $VLLM_LOG:"
            tail -30 "$VLLM_LOG" >&2
            return 1
        fi
        sleep 5
    done
    echo "!!! vLLM didn't become ready within 10 minutes. Tail of $VLLM_LOG:"
    tail -30 "$VLLM_LOG" >&2
    return 1
}

if ! is_vllm_up; then
    if [ "$AUTO_START_VLLM" = "1" ]; then
        start_vllm_in_bg || exit 1
        # Schedule a kill on script exit so we don't leave it dangling.
        trap '[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true' EXIT
    else
        cat >&2 <<EOF
!!! No vLLM server reachable at $BASE_URL.

Start one in another terminal:
    cd $REPO_ROOT
    source .venv/bin/activate
    set -a; source .env; set +a       # for HF_TOKEN
    vllm serve $MODEL --port 8000 --max-model-len 8192 --enforce-eager

… then re-run this script.

Alternatively, let this script auto-start it:
    AUTO_START_VLLM=1 bash eval/run_debug.sh
EOF
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

echo ">>> Running prime eval against $BASE_URL ($MODEL)"
prime eval run glyphbench \
  --provider vllm \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -k "$API_KEY_VAR" \
  -n "$NUM_EPISODES" \
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --sampling-args "$SAMPLING_ARGS" \
  -a "{\"task_id\": $TASK_IDS, \"num_episodes\": $NUM_EPISODES, \"n_frames\": $N_FRAMES, \"max_output_tokens\": $MAX_TOKENS}"
