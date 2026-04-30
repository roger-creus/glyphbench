#!/usr/bin/env bash
# One-time install of a user-space CUDA 12.6 toolkit for hosts that have
# no /usr/local/cuda and no sudo. flashinfer's gdn_prefill_sm90 kernel
# (used by Qwen3.5-4B's gated delta nets) JIT-compiles on first use and
# needs CUDA 12 nvcc + matching headers; the system's /usr/bin/nvcc is
# CUDA 11.5 (no sm_90 support).
#
# Why 12.6 and not 12.8: the GPU driver here (NVIDIA 565.57.01) caps out
# at CUDA 12.7. Emits host calls into TMA encoding APIs that only exist
# in CUDA 12.8 driver were rejected with error 999 at runtime.
#
# Output: $CUDA_HOME (default /home/roger/cuda-12.6) populated with
#   bin/{nvcc,ptxas,cudafe++,fatbinary,nvlink,bin2c}
#   bin/crt/{link.stub,prelink.stub}              (from system /usr/lib/nvidia-cuda-toolkit)
#   include/crt/{host_defines.h, ...}             (from CUDA 12.8 fallback — version-stable)
#   include/<runtime + cccl headers>              (from cu12 pip wheels)
#   lib64/libcudart.so{,.12} -> venv libcudart.12 (so JIT links against cu12 runtime,
#                                                  NOT system /usr/lib's libcudart 11.5,
#                                                  whose cudaGetDriverEntryPoint can't
#                                                  resolve cuTensorMapEncode* and so
#                                                  flashinfer's gdn TMA encode fails)
#   lib64/stubs/libcuda.so -> /usr/lib/.../libcuda.so.1
#   nvvm/{bin/cicc,lib64/libnvvm.so*}
#
# Sources:
#   - NVIDIA's nvidia/linux-64 conda channel for nvcc/nvvm binaries
#   - pip wheels for cccl + cuda-runtime headers (the 12.6 conda
#     cuda-cccl is a license-only stub; the actual headers ship in the
#     pip nvidia-cuda-cccl-cu12 wheel only)
#   - System /usr/lib/nvidia-cuda-toolkit for crt link stubs
#   - CUDA 12.8 install for crt header set (12.6 conda doesn't ship them)
#   - venv's nvidia-cuda-runtime-cu12 wheel for libcudart.so.12

set -euo pipefail

CUDA_HOME=${CUDA_HOME:-/home/roger/cuda-12.6}
CUDA_VERSION=${CUDA_VERSION:-12.6.85}
CUDART_VERSION=${CUDART_VERSION:-12.6.77}
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

extract_pip_headers() {
    local pkg=$1     # e.g. nvidia-cuda-cccl-cu12
    local ver=$2     # e.g. 12.6.77
    local subdir=$3  # subdir under nvidia/<subdir>/include in the wheel
    local whl="$WORK/${pkg}-${ver}.whl"
    local url
    url=$(python3 - "$pkg" "$ver" <<'PY'
import sys, urllib.request, json
pkg, ver = sys.argv[1:]
d = json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{pkg}/{ver}/json'))
for f in d['urls']:
    if 'manylinux2014_x86_64.manylinux_2_17_x86_64' in f['filename']:
        print(f['url']); break
PY
)
    if [ -z "$url" ]; then
        echo "ERROR: no x86_64 manylinux wheel for ${pkg}==${ver}" >&2
        return 1
    fi
    curl -sL -o "$whl" "$url"
    python3 - "$whl" "$subdir" "$CUDA_HOME/include" <<'PY'
import sys, zipfile, os
whl, subdir, out = sys.argv[1:]
prefix = f'nvidia/{subdir}/include/'
n_extracted = 0
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.startswith(prefix) and not n.endswith('/'):
            rel = n[len(prefix):]
            if rel == '__init__.py':
                continue
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as src, open(dst, 'wb') as f:
                f.write(src.read())
            n_extracted += 1
print(f'  extracted {n_extracted} headers from {whl}')
PY
}

echo "Extracting cuda-nvcc-tools (nvcc binary + ptxas + nvlink)..."
extract_conda "cuda-nvcc-tools-${CUDA_VERSION}-0"

echo "Extracting cuda-nvvm-tools (cicc — required by nvcc)..."
extract_conda "cuda-nvvm-tools-${CUDA_VERSION}-0"

echo "Extracting cuda-nvvm-impl (libnvvm.so)..."
extract_conda "cuda-nvvm-impl-${CUDA_VERSION}-0"

# Place binaries
cp -p "$WORK/cuda-nvcc-tools-${CUDA_VERSION}-0/bin/"* "$CUDA_HOME/bin/"
cp -p "$WORK/cuda-nvvm-tools-${CUDA_VERSION}-0/nvvm/bin/cicc" "$CUDA_HOME/nvvm/bin/"
cp -p "$WORK/cuda-nvvm-impl-${CUDA_VERSION}-0/nvvm/lib64/"libnvvm.so* "$CUDA_HOME/nvvm/lib64/"

echo "Extracting cccl headers (cub/thrust/libcudacxx) from pip wheel..."
extract_pip_headers "nvidia-cuda-cccl-cu12" "$CUDART_VERSION" "cuda_cccl"

echo "Extracting cuda-runtime headers (cuda_fp8.h, cuda_runtime.h, etc.) from pip wheel..."
extract_pip_headers "nvidia-cuda-runtime-cu12" "$CUDART_VERSION" "cuda_runtime"

# crt/host_defines.h (etc.) — defines __grid_constant__ and other
# host-side preprocessor macros. The 12.6 conda packages don't ship
# these; fall back to a CUDA 12.8 install if one already exists, or
# error out telling the user to install one manually.
CRT_FALLBACK=${CRT_FALLBACK:-/home/roger/cuda-12.8/include/crt}
if [ -d "$CRT_FALLBACK" ]; then
    echo "Copying crt/ headers from $CRT_FALLBACK..."
    cp -r "$CRT_FALLBACK" "$CUDA_HOME/include/"
else
    echo "WARNING: $CRT_FALLBACK missing — flashinfer JIT will fail to find" >&2
    echo "  __grid_constant__. Install a 12.x toolkit there or set CRT_FALLBACK." >&2
fi

# bin/crt/{link,prelink}.stub — needed for nvcc host linking. Pull from
# the system /usr/lib/nvidia-cuda-toolkit if available (Ubuntu ships
# them with the apt nvidia-cuda-toolkit package even on hosts where the
# rest of CUDA is unusable).
SYS_NVCC_CRT=${SYS_NVCC_CRT:-/usr/lib/nvidia-cuda-toolkit/bin/crt}
if [ -d "$SYS_NVCC_CRT" ]; then
    echo "Copying nvcc bin/crt/ stubs from $SYS_NVCC_CRT..."
    mkdir -p "$CUDA_HOME/bin/crt"
    cp -p "$SYS_NVCC_CRT/"* "$CUDA_HOME/bin/crt/"
fi

# Symlink venv's libcudart.so.12 into our toolkit lib64. Flashinfer's
# JIT links with `-L$cuda_home/lib64 -lcudart`. If lib64 has no
# libcudart, the linker resolves -lcudart against the SYSTEM
# /usr/lib/x86_64-linux-gnu/libcudart.so.11.5 (Ubuntu's CUDA 11.5),
# whose cudaGetDriverEntryPoint table doesn't know about
# cuTensorMapEncodeTiled — so cute's TMA descriptor wrapper returns
# CUDA_ERROR_UNKNOWN (999) and the gdn kernel can't run.
VENV_CUDART=${VENV_CUDART:-/home/roger/glyphbench/.worktrees/rl-pipeline-v1/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12}
if [ -f "$VENV_CUDART" ]; then
    echo "Symlinking $VENV_CUDART into $CUDA_HOME/lib64/..."
    ln -sf "$VENV_CUDART" "$CUDA_HOME/lib64/libcudart.so"
    ln -sf "$VENV_CUDART" "$CUDA_HOME/lib64/libcudart.so.12"
else
    echo "WARNING: $VENV_CUDART missing — set VENV_CUDART or run uv sync first" >&2
fi

# stubs/libcuda.so — nvcc's link line uses `-L$cuda_home/lib64/stubs`.
# The driver lib (libcuda.so.1) is provided by the host kernel module;
# the stub is just an import library so symbols resolve at link time.
mkdir -p "$CUDA_HOME/lib64/stubs"
SYS_LIBCUDA=${SYS_LIBCUDA:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}
if [ -f "$SYS_LIBCUDA" ]; then
    ln -sf "$SYS_LIBCUDA" "$CUDA_HOME/lib64/stubs/libcuda.so"
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
