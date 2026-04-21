"""Query each cluster's HF cache to discover which models are available.

Used by the experiment launcher to route (model, harness, suite) jobs only
to clusters that already have the model — avoiding silent failures where
HF_HUB_OFFLINE=1 combined with a missing snapshot would crash the job.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from clusters import ssh_run
from config import CLUSTERS, CLUSTER_NAMES


def _model_slug(model_id: str) -> str:
    # HF cache layout: <hf_cache>/hub/models--<owner>--<name>/snapshots/<sha>/
    return "models--" + model_id.replace("/", "--")


def _lookup_candidates(cluster: str) -> list[str]:
    """Possible HF cache roots for the given cluster.

    Mila has two-level nesting (`hub/hub/models--…`) because the configured
    HF_HOME = .../hub and HF auto-creates a hub/ subdir underneath. Other
    clusters are flat (`hub/models--…`).
    """
    cfg = CLUSTERS[cluster]
    return [f"{cfg['hf_cache']}/hub", f"{cfg['hf_cache']}/hub/hub"]


def cluster_has_model(cluster: str, model_id: str, timeout: int = 15) -> bool:
    """Return True iff the cluster has a usable snapshot of the model."""
    slug = _model_slug(model_id)
    for root in _lookup_candidates(cluster):
        rc, out, _ = ssh_run(cluster, f"ls {root}/{slug}/snapshots", timeout=timeout)
        if rc == 0 and out.strip():
            return True
    return False


def list_cached_models(cluster: str, timeout: int = 15) -> set[str]:
    """Return the set of HF repo ids cached on a cluster (best-effort)."""
    models: set[str] = set()
    for root in _lookup_candidates(cluster):
        rc, out, _ = ssh_run(cluster, f"ls {root}", timeout=timeout)
        if rc != 0:
            continue
        for line in out.splitlines():
            if line.startswith("models--"):
                # models--owner--name → owner/name (reverse the double-dash split)
                rest = line[len("models--"):]
                parts = rest.split("--", 1)
                if len(parts) == 2:
                    models.add(f"{parts[0]}/{parts[1]}")
    return models


def availability_matrix(
    clusters: list[str] | None = None,
) -> dict[str, set[str]]:
    """Return {cluster: {model_ids}} across the given clusters, in parallel."""
    clusters = clusters or CLUSTER_NAMES
    out: dict[str, set[str]] = {}
    with ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        futures = {pool.submit(list_cached_models, c): c for c in clusters}
        for f, c in futures.items():
            try:
                out[c] = f.result()
            except Exception:
                out[c] = set()
    return out
