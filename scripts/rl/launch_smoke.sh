#!/usr/bin/env bash
# Single-node smoke run — exercises the patched compute_advantages path
# and the custom loss without needing the full cluster.
#
# Requires 2× GPUs on this node.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$REPO_ROOT/outputs/smoke"

# We use the prime-rl `rl` entrypoint for the single-node case (vs. the
# multi-node manual launch for the real run). Override entrypoint via our
# patch module: it monkey-patches compute_advantages, then calls into prime-rl.
# Because the rl entrypoint also spawns inference, we manage everything from
# one process here.

# To make the rl entrypoint use our patched compute_advantages, we replicate
# the patch in a tiny shim module at smoke time.
cat > /tmp/glyphbench_smoke_shim.py <<'PY'
import prime_rl.orchestrator.advantage as _adv
from glyphbench.rl.advantage import compute_advantages_with_env_norm
_adv.compute_advantages = compute_advantages_with_env_norm
from prime_rl.entrypoints.rl import main
main()
PY

uv run --extra rl --extra eval python /tmp/glyphbench_smoke_shim.py \
    @ configs/rl/qwen35-4b-glyphbench/smoke.toml \
    --output-dir "$REPO_ROOT/outputs/smoke"
