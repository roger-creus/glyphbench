#!/usr/bin/env python3
"""GlyphBench multi-cluster experiment manager.

Usage:
    ./cm.py setup [--clusters C ...]
    ./cm.py submit [--clusters C ...] [--envs E ...] [--suites S ...] [--dry-run] [--pilot]
    ./cm.py status [--clusters C ...]
    ./cm.py pull [--clusters C ...]
    ./cm.py sync-code [--clusters C ...]
    ./cm.py push-sif [--clusters C ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clusters import (
    push_sif_to_cluster,
    ssh_run,
    ssh_run_all,
    submit_sbatch,
    sync_code_to_cluster,
    test_connections,
    rsync_from,
)
from config import (
    CLUSTER_NAMES,
    CLUSTERS,
    LOCAL_RESULTS_DIR,
    LOCAL_SIF_PATH,
    PROJECT_ROOT,
)
from jobs import generate_all_jobs, print_job_summary
from manifest import load_latest_manifest, save_manifest


def get_env_ids(args: argparse.Namespace) -> list[str]:
    """Get the list of env IDs to evaluate."""
    # Import here to avoid slow import at CLI startup
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import glyphbench  # noqa: F401
    from glyphbench.core import all_glyphbench_env_ids

    env_ids = all_glyphbench_env_ids()
    # Filter out dummy
    env_ids = [e for e in env_ids if "dummy" not in e]

    if hasattr(args, "envs") and args.envs:
        env_ids = args.envs
    elif hasattr(args, "suites") and args.suites:
        env_ids = [e for e in env_ids if any(s in e for s in args.suites)]

    return env_ids


# ===================================================================
# setup
# ===================================================================
def cmd_setup(args: argparse.Namespace) -> None:
    """Full setup: test connections, sync code, push SIF."""
    clusters = args.clusters or CLUSTER_NAMES

    print("\n[1/4] Testing SSH connections...")
    status = test_connections(clusters)
    reachable = [c for c, ok in status.items() if ok]
    if not reachable:
        print("ERROR: No reachable clusters.")
        sys.exit(1)

    print(f"\n[2/4] Creating directories on {len(reachable)} clusters...")
    for c in reachable:
        cfg = CLUSTERS[c]
        for d in [cfg["code_dir"], cfg["output_dir"], cfg["log_dir"]]:
            ssh_run(c, f"mkdir -p {d}")

    print(f"\n[3/4] Syncing code to {len(reachable)} clusters...")
    for c in reachable:
        sync_code_to_cluster(c, PROJECT_ROOT)

    print(f"\n[4/4] Pushing SIF to {len(reachable)} clusters...")
    if not LOCAL_SIF_PATH.exists():
        print(f"  ERROR: SIF not found at {LOCAL_SIF_PATH}")
        print("  Build it first: docker build -t glyphbench:latest . && "
              "apptainer build cluster_manager/glyphbench.sif docker-daemon://glyphbench:latest")
        sys.exit(1)
    for c in reachable:
        push_sif_to_cluster(c, LOCAL_SIF_PATH)

    print("\nSetup complete!")


# ===================================================================
# submit
# ===================================================================
def cmd_submit(args: argparse.Namespace) -> None:
    """Generate and submit SLURM jobs.

    Jobs are split per (model, harness, suite), not per env. --suites filters
    which suites to run; --pilot restricts to a minimal 1-suite smoke test.
    """
    clusters = args.clusters or CLUSTER_NAMES

    suites = args.suites if hasattr(args, "suites") and args.suites else None
    models = args.models if hasattr(args, "models") and args.models else None
    harnesses = (
        args.harnesses if hasattr(args, "harnesses") and args.harnesses else None
    )

    if args.pilot:
        # Smoke test: just minigrid suite, one model, one harness
        suites = ["minigrid"]
        if models is None:
            models = ["Qwen/Qwen3-0.6B"]
        if harnesses is None:
            harnesses = ["markov_zeroshot"]

    jobs = generate_all_jobs(
        models=models, harnesses=harnesses, suites=suites, clusters=clusters,
    )
    print_job_summary(jobs)

    if args.dry_run:
        print("\n[DRY RUN] Would submit the above jobs. Use without --dry-run to submit.")
        # Save first script for inspection
        if jobs:
            j = jobs[0]
            print(
                f"\nExample script ({j['model']}/{j['harness']}/{j['suite']} "
                f"on {j['cluster']}):"
            )
            print(j["script"][:1000])
        return

    print(f"\nSubmitting {len(jobs)} jobs...")
    submitted = []
    failed = 0
    for j in jobs:
        rc, result = submit_sbatch(j["cluster"], j["script"])
        if rc == 0:
            j["job_id"] = result
            submitted.append(j)
            print(f"  [{j['cluster']}] {j['job_name']}: job {result}")
        else:
            failed += 1
            print(f"  [{j['cluster']}] {j['job_name']}: FAILED ({result})")

    print(f"\nSubmitted: {len(submitted)}, Failed: {failed}")
    if submitted:
        save_manifest(jobs, submitted)


# ===================================================================
# status
# ===================================================================
def cmd_status(args: argparse.Namespace) -> None:
    """Check SLURM queue status across clusters."""
    clusters = args.clusters or CLUSTER_NAMES

    print("Querying job status...")
    for c in clusters:
        rc, out, err = ssh_run(c, "squeue -u rogercc")
        lines = out.split("\n")
        gb_jobs = [l for l in lines if "glyphbench" in l or "gb_" in l]
        running = sum(1 for l in gb_jobs if " R " in l)
        pending = sum(1 for l in gb_jobs if " PD" in l)
        print(f"  {c}: {running} running, {pending} pending, {len(gb_jobs)} total")
        if args.detailed:
            for l in gb_jobs[:20]:
                print(f"    {l.strip()}")
            if len(gb_jobs) > 20:
                print(f"    ... and {len(gb_jobs) - 20} more")


# ===================================================================
# pull
# ===================================================================
def cmd_pull(args: argparse.Namespace) -> None:
    """Pull results from all clusters."""
    clusters = args.clusters or CLUSTER_NAMES
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pulling results to {LOCAL_RESULTS_DIR}...")
    for c in clusters:
        output_dir = CLUSTERS[c]["output_dir"]
        rsync_from(c, output_dir, str(LOCAL_RESULTS_DIR), includes=["*.json"])

    # Count results
    json_files = list(LOCAL_RESULTS_DIR.rglob("*.json"))
    print(f"\nPulled {len(json_files)} result files.")


# ===================================================================
# sync-code
# ===================================================================
def cmd_sync_code(args: argparse.Namespace) -> None:
    """Sync code to clusters."""
    clusters = args.clusters or CLUSTER_NAMES
    print(f"Syncing code to {len(clusters)} clusters...")
    for c in clusters:
        sync_code_to_cluster(c, PROJECT_ROOT)
    print("Done.")


# ===================================================================
# push-sif
# ===================================================================
def cmd_push_sif(args: argparse.Namespace) -> None:
    """Push SIF container to clusters."""
    clusters = args.clusters or CLUSTER_NAMES
    if not LOCAL_SIF_PATH.exists():
        print(f"ERROR: SIF not found at {LOCAL_SIF_PATH}")
        sys.exit(1)
    print(f"Pushing {LOCAL_SIF_PATH.name} ({LOCAL_SIF_PATH.stat().st_size / 1e9:.1f}GB) "
          f"to {len(clusters)} clusters...")
    for c in clusters:
        push_sif_to_cluster(c, LOCAL_SIF_PATH)
    print("Done.")


# ===================================================================
# CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphBench cluster experiment manager")
    sub = parser.add_subparsers(dest="command")

    # setup
    p = sub.add_parser("setup", help="Full setup: test, sync, push SIF")
    p.add_argument("--clusters", nargs="*")

    # submit
    p = sub.add_parser("submit", help="Submit SLURM jobs")
    p.add_argument("--clusters", nargs="*")
    p.add_argument("--models", nargs="*", help="Restrict to these model IDs")
    p.add_argument("--harnesses", nargs="*", help="Restrict to these harness modes")
    p.add_argument("--envs", nargs="*")
    p.add_argument("--suites", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pilot", action="store_true", help="Submit a single-suite smoke test")

    # status
    p = sub.add_parser("status", help="Check job status")
    p.add_argument("--clusters", nargs="*")
    p.add_argument("--detailed", action="store_true")

    # pull
    p = sub.add_parser("pull", help="Pull results from clusters")
    p.add_argument("--clusters", nargs="*")

    # sync-code
    p = sub.add_parser("sync-code", help="Sync code to clusters")
    p.add_argument("--clusters", nargs="*")

    # push-sif
    p = sub.add_parser("push-sif", help="Push SIF to clusters")
    p.add_argument("--clusters", nargs="*")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {
        "setup": cmd_setup,
        "submit": cmd_submit,
        "status": cmd_status,
        "pull": cmd_pull,
        "sync-code": cmd_sync_code,
        "push-sif": cmd_push_sif,
    }[args.command](args)


if __name__ == "__main__":
    main()
