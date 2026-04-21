"""Central configuration for the GlyphBench multi-cluster experiment manager."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLUSTER_MANAGER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLUSTER_MANAGER_DIR.parent
GENERATED_SCRIPTS_DIR = CLUSTER_MANAGER_DIR / "generated_scripts"
MANIFESTS_DIR = CLUSTER_MANAGER_DIR / "manifests"
LOCAL_RESULTS_DIR = CLUSTER_MANAGER_DIR / "results"
LOCAL_SIF_PATH = CLUSTER_MANAGER_DIR / "glyphbench.sif"

# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
CLUSTERS: dict[str, dict] = {
    "rorqual": {
        "ssh_host": "rorqual-robot",
        "username": "rogercc",
        "account": "rrg-bengioy-ad",
        "scratch": "/scratch/rogercc",
        "code_dir": "/scratch/rogercc/glyphbench",
        "image_path": "/scratch/rogercc/images/glyphbench.sif",
        "output_dir": "/scratch/rogercc/glyphbench_runs",
        "hf_cache": "/scratch/rogercc/hf_cache",
        "log_dir": "/scratch/rogercc/glyphbench_runs/logs",
        "modules": "apptainer",
        "singularity_cmd": "apptainer",
        "available_gpus": {
            "h100": {"vram_gb": 80, "gres": "gpu:h100:1"},
            "h100_3g.40gb": {"vram_gb": 40, "gres": "gpu:h100_3g.40gb:1"},
        },
    },
    "narval": {
        "ssh_host": "narval-robot",
        "username": "rogercc",
        "account": "rrg-gberseth",
        "scratch": "/scratch/rogercc",
        "code_dir": "/scratch/rogercc/glyphbench",
        "image_path": "/scratch/rogercc/images/glyphbench.sif",
        "output_dir": "/scratch/rogercc/glyphbench_runs",
        "hf_cache": "/scratch/rogercc/hf_cache",
        "log_dir": "/scratch/rogercc/glyphbench_runs/logs",
        "modules": "apptainer",
        "singularity_cmd": "apptainer",
        "available_gpus": {
            "a100": {"vram_gb": 40, "gres": "gpu:a100:1"},
        },
    },
    "fir": {
        "ssh_host": "fir-robot",
        "username": "rogercc",
        "account": "rrg-bengioy-ad",
        "scratch": "/scratch/rogercc",
        "code_dir": "/scratch/rogercc/glyphbench",
        "image_path": "/scratch/rogercc/images/glyphbench.sif",
        "output_dir": "/scratch/rogercc/glyphbench_runs",
        "hf_cache": "/scratch/rogercc/hf_cache",
        "log_dir": "/scratch/rogercc/glyphbench_runs/logs",
        "modules": "apptainer",
        "singularity_cmd": "apptainer",
        "available_gpus": {
            "h100": {"vram_gb": 80, "gres": "gpu:h100:1"},
            "nvidia_h100_80gb_hbm3_3g.40gb": {"vram_gb": 40, "gres": "gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"},
        },
    },
    # Mila cluster (login node SSH, not robot)
    "mila": {
        "ssh_host": "mila",
        "username": "roger.creus-castanyer",
        "account": "rrg-bengioy-ad",  # not used on Mila (no account flag)
        "scratch": "/network/scratch/r/roger.creus-castanyer",
        "code_dir": "/network/scratch/r/roger.creus-castanyer/glyphbench",
        "image_path": "/network/scratch/r/roger.creus-castanyer/images/glyphbench.sif",
        "output_dir": "/network/scratch/r/roger.creus-castanyer/glyphbench_runs",
        "hf_cache": "/network/scratch/r/roger.creus-castanyer/hub",
        "log_dir": "/network/scratch/r/roger.creus-castanyer/glyphbench_runs/logs",
        "modules": "singularity",
        "singularity_cmd": "singularity",
        "available_gpus": {
            # lab-real partition
            "a6000": {"vram_gb": 48, "gres": "gpu:a6000:1", "partition": "lab-real", "nodelist": "cn-j001"},
            "a100": {"vram_gb": 48, "gres": "gpu:a100:1", "partition": "lab-real", "nodelist": "cn-k003,cn-k004"},
            # long partition (general)
            "long_1gpu": {"vram_gb": 48, "gres": "gpu:1", "partition": "long"},
        },
    },
}

CLUSTER_NAMES = list(CLUSTERS.keys())

# Mila-specific partitions
MILA_LAB_REAL = "lab-real"
MILA_LONG = "long"

# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    # Qwen3.5 family (dense, instruction-tuned)
    "Qwen/Qwen3.5-0.8B": {"min_vram_gb": 10, "tp": 1, "walltime": "3:59:59"},
    "Qwen/Qwen3.5-2B": {"min_vram_gb": 10, "tp": 1, "walltime": "5:59:59"},
    "Qwen/Qwen3.5-4B": {"min_vram_gb": 20, "tp": 1, "walltime": "5:59:59"},
    "Qwen/Qwen3.5-9B": {"min_vram_gb": 24, "tp": 1, "walltime": "11:59:59"},
    "Qwen/Qwen3.5-27B": {"min_vram_gb": 60, "tp": 2, "walltime": "23:59:59"},
    # Qwen3 family
    "Qwen/Qwen3-0.6B": {"min_vram_gb": 10, "tp": 1, "walltime": "3:59:59"},
    "Qwen/Qwen3-8B": {"min_vram_gb": 20, "tp": 1, "walltime": "11:59:59"},
    "Qwen/Qwen3-14B": {"min_vram_gb": 35, "tp": 1, "walltime": "11:59:59"},
    "Qwen/Qwen3-32B": {"min_vram_gb": 70, "tp": 2, "walltime": "23:59:59"},
    # DeepSeek R1 distills (reasoning-trained)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {"min_vram_gb": 10, "tp": 1, "walltime": "3:59:59"},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"min_vram_gb": 20, "tp": 1, "walltime": "11:59:59"},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {"min_vram_gb": 35, "tp": 1, "walltime": "11:59:59"},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {"min_vram_gb": 70, "tp": 2, "walltime": "23:59:59"},
    # Google Gemma 3
    "google/gemma-3-4b-it": {"min_vram_gb": 20, "tp": 1, "walltime": "5:59:59"},
    "google/gemma-3-12b-it": {"min_vram_gb": 30, "tp": 1, "walltime": "11:59:59"},
    "google/gemma-3-27b-it": {"min_vram_gb": 60, "tp": 2, "walltime": "23:59:59"},
    # Meta Llama
    "meta-llama/Llama-3.1-8B-Instruct": {"min_vram_gb": 20, "tp": 1, "walltime": "11:59:59"},
    # Mistral
    "mistralai/Mistral-7B-Instruct-v0.3": {"min_vram_gb": 20, "tp": 1, "walltime": "11:59:59"},
}

# ---------------------------------------------------------------------------
# Harness modes
# ---------------------------------------------------------------------------
HARNESS_MODES = [
    "markov_zeroshot",
    "markov_cot",
    "history_zeroshot",
    "history_cot",
]

# History length ablation values (for history modes only)
HISTORY_LENS = [1, 3, 5, 10]

# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------
EPISODES_PER_ENV = 10
MAX_TURNS = 200
BATCH_SIZE = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 16384  # generous thinking budget
MAX_MODEL_LEN = 32768
DTYPE = "bfloat16"

# ---------------------------------------------------------------------------
# Game suites (for smart job splitting: 1 job per suite per model per harness)
# ---------------------------------------------------------------------------
SUITES = ["minigrid", "minihack", "atari", "procgen", "craftax", "classics"]

# Resource profile per job
RESOURCE_PROFILE = {
    "cpus": 8,
    "mem": "48G",
    "walltime": "11:59:59",  # 12h max (some suites have many envs)
}

# ---------------------------------------------------------------------------
# Code sync excludes
# ---------------------------------------------------------------------------
CODE_SYNC_EXCLUDES = [
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "runs/",
    "results/",
    "specs/",
    "plans/",
    ".claude/",
    "*.egg-info",
    "uv.lock",
    "*.zip",
    ".env",
    "cluster_manager/glyphbench.sif",
    "cluster_manager/results/",
]
