# GlyphBench — verifiers + prime-rl container
#
# Build:   docker build -t glyphbench:latest .
# SIF:     bash scripts/build_sif.sh
#
# Run eval inside container (with model weights mounted):
#   apptainer run --nv --bind $HF_HOME:/root/.cache/huggingface \
#     glyphbench.sif bash eval/run_debug.sh

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# uv (Python & Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && cp /root/.local/bin/uv /usr/local/bin/uv \
  && cp /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /opt/glyphbench

# Install dependencies first (cached layer). We copy just the files required
# by the build backend (pyproject.toml refers to README.md in [project.readme],
# and hatchling builds the wheel from src/glyphbench) so this layer is cached
# independent of tests/docs/configs edits.
COPY pyproject.toml uv.lock README.md /opt/glyphbench/
COPY src /opt/glyphbench/src
RUN uv python install 3.12 \
  && uv sync --frozen --extra eval

# Copy the rest of the source and re-sync (no-op if nothing changed).
COPY . /opt/glyphbench
RUN uv sync --frozen --extra eval

# Optional RL extra (prime-rl + flash-attn). Heavy; install at run time if needed.
# RUN uv sync --frozen --extra eval --extra rl

ENV PATH="/opt/glyphbench/.venv/bin:${PATH}"

# uv installs the managed Python under /root/.local/share/uv/...; the venv
# at /opt/glyphbench/.venv/bin/python symlinks to it. Older Singularity
# (e.g. 3.7 on Mila) runs as the host user rather than fakeroot, so it
# can't stat anything under /root unless /root is world-readable. Loosen
# perms after install so non-root execs of the venv interpreter succeed.
RUN chmod -R a+rX /root || true

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
