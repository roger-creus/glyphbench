#!/usr/bin/env bash
# One-time install of a user-space CUDA 12.8 toolkit for hosts that have
# no /usr/local/cuda and no sudo. flashinfer's gdn_prefill_sm90 kernel
# (used by Qwen3.5-4B's gated delta nets) JIT-compiles on first use and
# needs CUDA 12 nvcc + matching headers; the system's /usr/bin/nvcc is
# CUDA 11.5 (no sm_90 support).
#
# Output: $CUDA_HOME (default /home/roger/cuda-12.8) populated with
#   bin/{nvcc,ptxas,cudafe++,fatbinary,nvlink,bin2c}
#   nvvm/{bin/cicc,lib64/libnvvm.so*}
#   include/* (cuda runtime, cuda_fp8.h, cccl, etc.)
#
# Sources: NVIDIA's nvidia/linux-64 conda channel + the cuda runtime
# headers shipped in nvidia-cuda-runtime-cu12 (already installed by
# `uv sync --extra rl`).

set -euo pipefail

CUDA_HOME=${CUDA_HOME:-/home/roger/cuda-12.8}
CUDA_VERSION=${CUDA_VERSION:-12.8.93}
CUDART_VERSION=${CUDART_VERSION:-12.8.90}
WORK=/tmp/cuda-install-$$

mkdir -p "$WORK" "$CUDA_HOME/bin" "$CUDA_HOME/nvvm/bin" \
    "$CUDA_HOME/nvvm/lib64" "$CUDA_HOME/include" "$CUDA_HOME/lib64"

extract_conda() {
    local name=$1
    local out="$WORK/${name}.conda"
    curl -sL -o "$out" "https://conda.anaconda.org/nvidia/linux-64/${name}.conda"
    python3 - "$out" "$WORK/$name" <<'PY'
import sys, zipfile, subprocess
src, dst = sys.argv[1:]
with zipfile.ZipFile(src) as z:
    pkg = [n for n in z.namelist() if n.startswith('pkg-')][0]
    with z.open(pkg) as f, open(f'{dst}.tar.zst', 'wb') as out:
        out.write(f.read())
subprocess.run(['zstd', '-d', '-q', '-f', f'{dst}.tar.zst', '-o', f'{dst}.tar'])
import os
os.makedirs(dst, exist_ok=True)
subprocess.run(['tar', '-xf', f'{dst}.tar', '-C', dst])
PY
}

echo "Extracting cuda-nvcc-tools (nvcc binary + ptxas + nvlink)..."
extract_conda "cuda-nvcc-tools-${CUDA_VERSION}-0"

echo "Extracting cuda-nvvm-tools (cicc — required by nvcc)..."
extract_conda "cuda-nvvm-tools-${CUDA_VERSION}-0"

echo "Extracting cuda-nvvm-impl (libnvvm.so)..."
extract_conda "cuda-nvvm-impl-${CUDA_VERSION}-0"

echo "Extracting cuda-cccl (cub/thrust/cuda headers)..."
extract_conda "cuda-cccl-${CUDART_VERSION}-0"

# Place binaries
cp -p "$WORK/cuda-nvcc-tools-${CUDA_VERSION}-0/bin/"* "$CUDA_HOME/bin/"
cp -p "$WORK/cuda-nvvm-tools-${CUDA_VERSION}-0/nvvm/bin/cicc" "$CUDA_HOME/nvvm/bin/"
cp -p "$WORK/cuda-nvvm-impl-${CUDA_VERSION}-0/nvvm/lib64/"libnvvm.so* "$CUDA_HOME/nvvm/lib64/"
cp -rT "$WORK/cuda-cccl-${CUDART_VERSION}-0/targets/x86_64-linux/include" "$CUDA_HOME/include"

# CUDA runtime headers (cuda_fp8.h, cuda_runtime.h, etc.) live in the
# pip nvidia-cuda-runtime-cu12 wheel that uv already installed.
RUNTIME_INC="$(dirname "$(realpath "$(dirname "$0")/../..")")/.worktrees/rl-pipeline-v1/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/include"
if [ ! -d "$RUNTIME_INC" ]; then
    # Fallback: search venvs under the worktree
    RUNTIME_INC="$(find "$(dirname "$0")/../../" -path '*/nvidia/cuda_runtime/include' -type d 2>/dev/null | head -1)"
fi
if [ -d "$RUNTIME_INC" ]; then
    cp -rT "$RUNTIME_INC" "$CUDA_HOME/include"
else
    echo "WARNING: nvidia/cuda_runtime/include not found — install nvidia-cuda-runtime-cu12 via uv first" >&2
fi

# Smoke-test
"$CUDA_HOME/bin/nvcc" --version
echo
echo "  CUDA toolkit ready at: $CUDA_HOME"
echo "  Use with: export CUDA_HOME=$CUDA_HOME && export PATH=\$CUDA_HOME/bin:\$PATH"
echo
echo "Cleanup work dir:"
rm -rf "$WORK"
echo "  removed $WORK"
