#!/usr/bin/env python3
"""Generate and submit sbatch jobs that pre-download HF models into each cluster's cache.

One small CPU+internet job per (cluster, model). Jobs run in the same SIF container
so `huggingface_hub.snapshot_download` is guaranteed to work.

Usage:
    python3 download_models.py                       # all missing models, all clusters
    python3 download_models.py --clusters mila rorqual --models Qwen/Qwen3-0.6B
    python3 download_models.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clusters import ssh_run, submit_sbatch
from config import CLUSTERS, CLUSTER_NAMES, MODELS


DOWNLOAD_SBATCH = r"""#!/bin/bash
{sbatch_account}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:59:59
#SBATCH --job-name=hf-dl-{model_short}
#SBATCH --output={log_dir}/hfdl-%x-%j.out
#SBATCH --error={log_dir}/hfdl-%x-%j.err
{sbatch_partition}

{module_load}

export HF_HOME={hf_cache}
export TRANSFORMERS_CACHE={hf_cache}/hub
export HUGGINGFACE_HUB_CACHE={hf_cache}/hub

{singularity_cmd} exec \
    --bind {hf_cache}:/hf_cache \
    --env HF_HOME=/hf_cache \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    {image_path} \
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='{model_id}',
    allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.model'],
    local_dir_use_symlinks=False,
)
print('Downloaded {model_id}')
"
"""


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--").replace(".", "")[:40]


def check_cached(cluster: str, model_id: str) -> bool:
    """Best-effort check of whether the model is already cached."""
    hf_cache = CLUSTERS[cluster]["hf_cache"]
    slug = "models--" + model_id.replace("/", "--")
    rc, out, _ = ssh_run(cluster, f"ls {hf_cache}/hub/{slug}", timeout=15)
    if rc == 0 and out.strip():
        return True
    # Mila sometimes has an extra hub/ nesting
    rc, out, _ = ssh_run(cluster, f"ls {hf_cache}/hub/hub/{slug}", timeout=15)
    return rc == 0 and bool(out.strip())


def generate_download_job(cluster: str, model_id: str) -> str:
    cfg = CLUSTERS[cluster]
    account = cfg.get("account", "")
    sbatch_account = f"#SBATCH --account={account}" if account and cluster != "mila" else ""

    # For Mila: pick a CPU-only partition if available; else use long
    if cluster == "mila":
        sbatch_partition = "#SBATCH --partition=long-cpu"
    else:
        sbatch_partition = ""

    module_load = f"module load {cfg['modules']}" if cfg.get("modules") else ""

    return DOWNLOAD_SBATCH.format(
        sbatch_account=sbatch_account,
        sbatch_partition=sbatch_partition,
        module_load=module_load,
        log_dir=cfg["log_dir"],
        hf_cache=cfg["hf_cache"],
        image_path=cfg["image_path"],
        singularity_cmd=cfg["singularity_cmd"],
        model_id=model_id,
        model_short=model_slug(model_id)[:20],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs="*")
    parser.add_argument("--models", nargs="*", help="Model IDs (defaults to all in config.MODELS)")
    parser.add_argument("--skip-cache-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    clusters = args.clusters or CLUSTER_NAMES
    models = args.models or list(MODELS.keys())

    jobs = []
    for cluster in clusters:
        for model_id in models:
            if not args.force and not args.skip_cache_check:
                if check_cached(cluster, model_id):
                    print(f"  [{cluster}] {model_id}: already cached, skip")
                    continue
            script = generate_download_job(cluster, model_id)
            jobs.append((cluster, model_id, script))
            print(f"  [{cluster}] {model_id}: will submit")

    print(f"\nTotal download jobs: {len(jobs)}")
    if args.dry_run or not jobs:
        return

    submitted = 0
    for cluster, model_id, script in jobs:
        rc, jid = submit_sbatch(cluster, script)
        if rc == 0:
            print(f"  [{cluster}] {model_id}: job {jid}")
            submitted += 1
        else:
            print(f"  [{cluster}] {model_id}: FAILED ({jid})")
    print(f"\nSubmitted {submitted}/{len(jobs)}")


if __name__ == "__main__":
    main()
