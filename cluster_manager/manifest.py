"""Job manifest: track submitted jobs for status and result pulling."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import MANIFESTS_DIR


def save_manifest(jobs: list[dict], submitted_jobs: list[dict]) -> Path:
    """Save a manifest of submitted jobs."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MANIFESTS_DIR / f"manifest_{ts}.json"
    manifest = {
        "timestamp": ts,
        "total_jobs": len(jobs),
        "submitted": len(submitted_jobs),
        "jobs": submitted_jobs,
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {path}")
    return path


def load_latest_manifest() -> dict | None:
    """Load the most recent manifest."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifests = sorted(MANIFESTS_DIR.glob("manifest_*.json"))
    if not manifests:
        return None
    with manifests[-1].open() as f:
        return json.load(f)


def list_manifests() -> list[Path]:
    """List all manifest files."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(MANIFESTS_DIR.glob("manifest_*.json"))
