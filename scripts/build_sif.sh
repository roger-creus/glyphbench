#!/usr/bin/env bash
# Build the GlyphBench container and convert to singularity/apptainer SIF.
# Overwrites any existing IMAGE or SIF.
set -euo pipefail
IMAGE=${IMAGE:-glyphbench:latest}
SIF=${SIF:-glyphbench.sif}

echo ">>> Docker build $IMAGE"
docker build -t "$IMAGE" .

echo ">>> Apptainer build $SIF"
apptainer build --force "$SIF" "docker-daemon://$IMAGE"

echo ">>> Done: $SIF"
ls -lh "$SIF"
