"""Job generation: sbatch scripts and workload distribution.

Jobs are split by (model, harness_mode, suite) — one SLURM job runs all envs
in a suite for a given model+harness combo. This gives ~72 jobs for the main
experiment (3 models x 4 modes x 6 suites).
"""

from __future__ import annotations

from config import (
    BATCH_SIZE,
    CLUSTERS,
    CLUSTER_NAMES,
    DTYPE,
    EPISODES_PER_ENV,
    HARNESS_MODES,
    HISTORY_LENS,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MAX_TURNS,
    MODELS,
    RESOURCE_PROFILE,
    SUITES,
    TEMPERATURE,
)


def select_gpu(cluster: str, model_id: str) -> str:
    """Select the smallest GPU with enough VRAM for the model."""
    min_vram = MODELS[model_id]["min_vram_gb"]
    gpus = CLUSTERS[cluster]["available_gpus"]
    candidates = [
        (name, info) for name, info in gpus.items()
        if info["vram_gb"] >= min_vram
    ]
    candidates.sort(key=lambda x: x[1]["vram_gb"])
    if not candidates:
        candidates = sorted(gpus.items(), key=lambda x: x[1]["vram_gb"], reverse=True)
    return candidates[0][1]["gres"]


SBATCH_TEMPLATE = r"""#!/bin/bash
{sbatch_account}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres={gres}
{sbatch_partition}
{sbatch_nodelist}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/slurm-{job_name}-%j.out
#SBATCH --error={log_dir}/slurm-{job_name}-%j.err

{module_load}

export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torch_cache"
export VLLM_COMPILE_CACHE_DIR="$SLURM_TMPDIR/vllm_compile_cache"
export PYTHONUNBUFFERED=1
export HF_HOME={hf_cache}
export TRANSFORMERS_CACHE={hf_cache}
export HUGGINGFACE_HUB_CACHE={hf_cache}

echo "=== GlyphBench Eval: $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "Model: {model_id} | Harness: {harness} | Suite: {suite} | Started: $(date)"

SIF={image_path}

mkdir -p {output_dir}

{singularity_cmd} exec --nv \
    --bind {code_dir}:/src \
    --bind {output_dir}:/results \
    --bind {hf_cache}:/hf_cache \
    --env HF_HOME=/hf_cache \
    --env TRANSFORMERS_CACHE=/hf_cache/hub \
    --env HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_NO_USAGE_STATS=1 \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache \
    --env VLLM_COMPILE_CACHE_DIR=/tmp/vllm_compile_cache \
    --env PYTHONPATH=/src/src \
    --env CUDA_MODULE_LOADING=LAZY \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env PYTHONUNBUFFERED=1 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --workdir /src \
    $SIF \
    python3 /src/eval/run_eval.py \
        --model {model_id} \
        --harness {harness} \
        --history-len {history_len} \
        --episodes {episodes} \
        --max-turns {max_turns} \
        --batch-size {batch_size} \
        --max-new-tokens {max_new_tokens} \
        --max-model-len {max_model_len} \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --dtype {dtype} \
        --temperature {temperature} \
        --tensor-parallel-size {tp} \
        --suites {suite} \
        --output /results/

_exit=$?
echo "=== Finished: $(date), exit=$_exit ==="
exit $_exit
"""


def _select_gpu_info(cluster: str, model_id: str) -> dict:
    """Select best GPU and return its full info dict."""
    min_vram = MODELS[model_id]["min_vram_gb"]
    gpus = CLUSTERS[cluster]["available_gpus"]
    candidates = [
        (name, info) for name, info in gpus.items()
        if info["vram_gb"] >= min_vram
    ]
    candidates.sort(key=lambda x: x[1]["vram_gb"])
    if not candidates:
        candidates = sorted(gpus.items(), key=lambda x: x[1]["vram_gb"], reverse=True)
    return candidates[0][1]


def generate_job(
    model_id: str,
    harness: str,
    suite: str,
    cluster: str,
    history_len: int = 5,
) -> dict:
    """Generate a single SLURM job for (model, harness, suite)."""
    cfg = CLUSTERS[cluster]
    gpu_info = _select_gpu_info(cluster, model_id)
    gres = gpu_info["gres"]
    tp = MODELS[model_id].get("tp", 1)

    # Multi-GPU: request N GPUs for tensor parallelism
    if tp > 1:
        # e.g., gpu:a6000:2 for TP=2
        parts = gres.rsplit(":", 1)
        gres = f"{parts[0]}:{tp}"

    model_short = model_id.split("/")[-1].lower().replace(".", "")[:15]
    harness_short = harness.replace("_", "")[:8]
    job_name = f"gb-{model_short}-{harness_short}-{suite}"[:40]

    walltime = MODELS[model_id].get("walltime", RESOURCE_PROFILE["walltime"])

    # Cluster-specific SBATCH directives
    account = cfg.get("account", "")
    sbatch_account = f"#SBATCH --account={account}" if account and cluster != "mila" else ""
    partition = gpu_info.get("partition", "")
    sbatch_partition = f"#SBATCH --partition={partition}" if partition else ""
    nodelist = gpu_info.get("nodelist", "")
    sbatch_nodelist = f"#SBATCH --nodelist={nodelist}" if nodelist else ""
    modules = cfg.get("modules", "")
    module_load = f"module load {modules}" if modules else ""

    # CUDA_VISIBLE_DEVICES for TP
    cuda_devices = ",".join(str(i) for i in range(tp))

    script = SBATCH_TEMPLATE.format(
        sbatch_account=sbatch_account,
        cpus=RESOURCE_PROFILE["cpus"],
        gres=gres,
        sbatch_partition=sbatch_partition,
        sbatch_nodelist=sbatch_nodelist,
        mem=RESOURCE_PROFILE["mem"],
        walltime=walltime,
        job_name=job_name,
        log_dir=cfg["log_dir"],
        module_load=module_load,
        hf_cache=cfg["hf_cache"],
        image_path=cfg["image_path"],
        code_dir=cfg["code_dir"],
        output_dir=cfg["output_dir"],
        singularity_cmd=cfg["singularity_cmd"],
        model_id=model_id,
        harness=harness,
        history_len=history_len,
        suite=suite,
        episodes=EPISODES_PER_ENV,
        max_turns=MAX_TURNS,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        temperature=TEMPERATURE,
        tp=tp,
        cluster=cluster,
    ).replace(
        "--env CUDA_VISIBLE_DEVICES=0", f"--env CUDA_VISIBLE_DEVICES={cuda_devices}"
    )

    return {
        "model": model_id,
        "harness": harness,
        "suite": suite,
        "cluster": cluster,
        "job_name": job_name,
        "gres": gres,
        "script": script,
    }


def generate_all_jobs(
    models: list[str] | None = None,
    harnesses: list[str] | None = None,
    suites: list[str] | None = None,
    clusters: list[str] | None = None,
    history_len: int = 5,
    cache_map: dict[str, set[str]] | None = None,
) -> list[dict]:
    """Generate the full experiment matrix, routing each (model,harness,suite)
    job to a cluster that has the model cached.

    If `cache_map` is provided (e.g. from `model_cache.availability_matrix`),
    jobs are only placed on clusters that already have the model. Among
    eligible clusters we use a round-robin that is scoped *per model* so that
    every model's suites are balanced across its eligible clusters.

    If `cache_map` is None, the old behaviour (round-robin across all clusters)
    is used — in which case callers must pre-verify cache availability.

    Raises:
        ValueError: if `cache_map` is given and a requested model is not
            cached on any of the candidate clusters.
    """
    models = models or list(MODELS.keys())
    harnesses = harnesses or HARNESS_MODES
    suites = suites or SUITES
    clusters = clusters or CLUSTER_NAMES

    jobs: list[dict] = []

    for model in models:
        if cache_map is not None:
            eligible = [c for c in clusters if model in cache_map.get(c, set())]
            if not eligible:
                raise ValueError(
                    f"Model '{model}' is not cached on any of {clusters}. "
                    f"Download it first (python3 download_models.py --models {model})."
                )
        else:
            eligible = clusters

        # Per-model round-robin across eligible clusters so each model's
        # harness×suite jobs are balanced across its candidate clusters.
        idx = 0
        for harness in harnesses:
            for suite in suites:
                cluster = eligible[idx % len(eligible)]
                idx += 1
                jobs.append(generate_job(
                    model, harness, suite, cluster,
                    history_len=history_len,
                ))
    return jobs


def summarize_availability(
    cache_map: dict[str, set[str]],
    models: list[str] | None = None,
    clusters: list[str] | None = None,
) -> None:
    """Pretty-print a model-availability matrix across clusters."""
    models = models or list(MODELS.keys())
    clusters = clusters or CLUSTER_NAMES
    col_w = max(len(m) for m in models) + 2
    header = "Model".ljust(col_w) + "  " + "  ".join(c[:8].center(8) for c in clusters)
    print(header)
    print("-" * len(header))
    for m in models:
        row = m.ljust(col_w)
        for c in clusters:
            mark = "   ✓   " if m in cache_map.get(c, set()) else "   -   "
            row += f"  {mark}"
        print(row)
    print()
    cached_counts = {c: len(cache_map.get(c, set()) & set(models)) for c in clusters}
    print("Cached-per-cluster:", cached_counts)


def generate_history_ablation_jobs(
    models: list[str] | None = None,
    suites: list[str] | None = None,
    clusters: list[str] | None = None,
) -> list[dict]:
    """Generate history length ablation: models x history_modes x N_values x suites."""
    models = models or list(MODELS.keys())
    suites = suites or SUITES
    clusters = clusters or CLUSTER_NAMES

    jobs = []
    idx = 0
    for model in models:
        for harness in ["history_zeroshot", "history_cot"]:
            for n in HISTORY_LENS:
                for suite in suites:
                    cluster = clusters[idx % len(clusters)]
                    idx += 1
                    jobs.append(generate_job(
                        model, harness, suite, cluster,
                        history_len=n,
                    ))
    return jobs


def print_job_summary(jobs: list[dict]) -> None:
    """Print a summary of generated jobs."""
    print(f"\nTotal jobs: {len(jobs)}")
    by_cluster: dict[str, int] = {}
    by_model: dict[str, int] = {}
    by_harness: dict[str, int] = {}
    for j in jobs:
        by_cluster[j["cluster"]] = by_cluster.get(j["cluster"], 0) + 1
        by_model[j["model"]] = by_model.get(j["model"], 0) + 1
        by_harness[j["harness"]] = by_harness.get(j["harness"], 0) + 1
    print("  By cluster:", dict(sorted(by_cluster.items())))
    print("  By model:", dict(sorted(by_model.items())))
    print("  By harness:", dict(sorted(by_harness.items())))
