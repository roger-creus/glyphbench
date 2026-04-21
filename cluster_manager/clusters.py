"""SSH and rsync operations for remote clusters (via robot nodes)."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import CLUSTERS, CLUSTER_NAMES, CODE_SYNC_EXCLUDES


def ssh_run(
    cluster: str,
    command: str,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Run a command on a cluster via SSH (robot node).

    Returns (returncode, stdout, stderr).
    """
    host = CLUSTERS[cluster]["ssh_host"]
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def ssh_run_all(
    command: str,
    clusters: list[str] | None = None,
    timeout: int = 300,
) -> dict[str, tuple[int, str, str]]:
    """Run a command on multiple clusters in parallel."""
    clusters = clusters or CLUSTER_NAMES
    results: dict[str, tuple[int, str, str]] = {}
    with ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        futures = {
            pool.submit(ssh_run, c, command, timeout): c for c in clusters
        }
        for future in as_completed(futures):
            cluster_name = futures[future]
            try:
                results[cluster_name] = future.result()
            except Exception as e:
                results[cluster_name] = (1, "", str(e))
    return results


def rsync_to(
    cluster: str,
    local_path: str,
    remote_path: str,
    excludes: list[str] | None = None,
    delete: bool = False,
) -> int:
    """rsync local files to a cluster (via robot node)."""
    host = CLUSTERS[cluster]["ssh_host"]
    user = CLUSTERS[cluster]["username"]
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", "ssh -o ConnectTimeout=15",
    ]
    if delete:
        cmd.append("--delete")
    for exc in (excludes or []):
        cmd.extend(["--exclude", exc])
    # Ensure trailing slash on local path for content sync
    local = local_path.rstrip("/") + "/"
    cmd.extend([local, f"{host}:{remote_path}/"])
    print(f"  [{cluster}] rsync → {remote_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{cluster}] rsync FAILED: {result.stderr[:200]}", file=sys.stderr)
    return result.returncode


def rsync_from(
    cluster: str,
    remote_path: str,
    local_path: str,
    includes: list[str] | None = None,
) -> int:
    """rsync files from a cluster to local."""
    host = CLUSTERS[cluster]["ssh_host"]
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", "ssh -o ConnectTimeout=15",
    ]
    if includes:
        cmd.extend(["--include=*/"])
        for inc in includes:
            cmd.extend(["--include", inc])
        cmd.extend(["--exclude=*"])
    cmd.extend([f"{host}:{remote_path}/", local_path + "/"])
    print(f"  [{cluster}] rsync ← {remote_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{cluster}] rsync FAILED: {result.stderr[:200]}", file=sys.stderr)
    return result.returncode


def sync_code_to_cluster(cluster: str, project_root: Path) -> int:
    """Sync project code to a cluster."""
    code_dir = CLUSTERS[cluster]["code_dir"]
    return rsync_to(cluster, str(project_root), code_dir, excludes=CODE_SYNC_EXCLUDES)


def push_sif_to_cluster(cluster: str, local_sif: Path) -> int:
    """Push the SIF container image to a cluster."""
    remote_dir = f"{CLUSTERS[cluster]['scratch']}/images"
    # Create remote dir first
    ssh_run(cluster, f"mkdir -p {remote_dir}")
    remote_path = f"{remote_dir}/{local_sif.name}"
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", "ssh -o ConnectTimeout=15",
        str(local_sif),
        f"{CLUSTERS[cluster]['ssh_host']}:{remote_path}",
    ]
    print(f"  [{cluster}] pushing SIF → {remote_path}")
    result = subprocess.run(cmd)
    return result.returncode


def test_connections(clusters: list[str] | None = None) -> dict[str, bool]:
    """Test SSH connectivity to robot nodes."""
    clusters = clusters or CLUSTER_NAMES
    results = {}
    for c in clusters:
        rc, out, err = ssh_run(c, "sbatch --version", timeout=10)
        results[c] = rc == 0
        status = "OK" if rc == 0 else "FAIL"
        print(f"  {c}: {status}")
    return results


def submit_sbatch(cluster: str, script_content: str) -> tuple[int, str]:
    """Submit an sbatch script via heredoc through SSH.

    Returns (returncode, job_id or error message).
    """
    host = CLUSTERS[cluster]["ssh_host"]
    # Write script to a temp file on the cluster, then sbatch it
    # Robot nodes typically allow sbatch with stdin
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", host, "sbatch"],
        input=script_content,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        # Parse job ID from "Submitted batch job 12345"
        parts = result.stdout.strip().split()
        job_id = parts[-1] if parts else "unknown"
        return 0, job_id
    else:
        return result.returncode, result.stderr.strip()
