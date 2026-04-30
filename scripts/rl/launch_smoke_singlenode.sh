#!/usr/bin/env bash
# Single-node smoke — runs trainer + orchestrator + 7-way vLLM inference
# all on sf-node-1. Used while inter-node ports (8000, 29501) are blocked
# by the cluster firewall. Once those open, use launch_smoke_multinode.sh
# instead.
#
# This goes through prime-rl's `rl` entrypoint which spawns inference,
# orchestrator, and trainer as child processes on the same host. We
# inject our compute_advantages monkey-patch via a small shim before
# calling the rl main.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR=${OUTPUT_DIR:-/home/roger/glyphbench/outputs/qwen35-4b-glyphbench-smoke-singlenode}
mkdir -p "$OUTPUT_DIR"

# Shim: monkey-patch then call into prime-rl's rl entrypoint.
SHIM=/tmp/glyphbench_smoke_shim.py
cat > "$SHIM" <<'PY'
import prime_rl.orchestrator.advantage as _adv
from glyphbench.rl.advantage import compute_advantages_with_env_norm
_adv.compute_advantages = compute_advantages_with_env_norm

# Also patch the orchestrator module's local binding (in case it was
# imported before our patch — belt and braces).
try:
    import prime_rl.orchestrator.orchestrator as _orch
    if _orch.compute_advantages is not compute_advantages_with_env_norm:
        _orch.compute_advantages = compute_advantages_with_env_norm
except Exception:
    pass

from prime_rl.entrypoints.rl import main
main()
PY

echo "[$(hostname)] launching single-node smoke (rl entrypoint)"
echo "  config: configs/rl/qwen35-4b-glyphbench/smoke-singlenode.toml"
echo "  output: $OUTPUT_DIR"

uv run --extra rl --extra eval python "$SHIM" \
    @ configs/rl/qwen35-4b-glyphbench/smoke-singlenode.toml \
    --output-dir "$OUTPUT_DIR"
