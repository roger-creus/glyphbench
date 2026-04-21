#!/usr/bin/env python
"""Upload generated assets (GIFs, etc.) to the GlyphBench HF dataset repo.

Usage:
    uv run python scripts/upload_assets.py                          # all gifs
    uv run python scripts/upload_assets.py --src docs/leaderboard/gifs --dst gifs
    uv run python scripts/upload_assets.py --repo owner/name --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="anon-paper-submission/glyphbench-assets",
                        help="HF dataset repo id (owner/name)")
    parser.add_argument("--src", type=Path, default=Path("docs/leaderboard/gifs"),
                        help="Local dir to upload")
    parser.add_argument("--dst", default="gifs",
                        help="Destination path within the dataset repo")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--commit-message", default="Update GlyphBench assets")
    parser.add_argument("--manifest", type=Path, default=Path("docs/leaderboard/gifs_manifest.json"),
                        help="Write a manifest of uploaded GIFs (consumed by gallery.html)")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source dir not found: {args.src}")

    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    api = HfApi()
    files = sorted(p for p in args.src.iterdir() if p.is_file())
    print(f"Uploading {len(files)} files from {args.src} -> {args.repo}/{args.dst}")
    api.upload_folder(
        folder_path=str(args.src),
        path_in_repo=args.dst,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=args.commit_message,
    )

    # Emit a manifest that gallery.html consumes.
    # Our slug scheme: "<env_id with / replaced by __>.gif".
    entries = []
    for p in files:
        if p.suffix != ".gif":
            continue
        slug = p.stem
        env_id = slug.replace("__", "/", 1)
        entries.append({"env_id": env_id, "slug": slug})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entries, indent=2))
    print(f"Wrote manifest ({len(entries)} entries) to {args.manifest}")

    print(
        f"\nDone. Public URL pattern:\n"
        f"  https://huggingface.co/datasets/{args.repo}/resolve/main/{args.dst}/<filename>"
    )


if __name__ == "__main__":
    main()
