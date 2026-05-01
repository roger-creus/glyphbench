"""Pre-populate the reward audit CSV with one row per registered env."""
from __future__ import annotations

import csv
import pathlib

import glyphbench  # noqa: F401 — registers all envs
from glyphbench.core.task_selection import list_task_ids

OUT = pathlib.Path("docs/superpowers/specs/2026-05-01-reward-audit.csv")


def _suite(env_id: str) -> str:
    return env_id.split("/", 1)[1].split("-", 1)[0]


def main() -> int:
    rows = [
        {
            "env_id": eid,
            "suite": _suite(eid),
            "pattern": "",
            "reward_formula": "",
            "notes": "",
            "status": "",
        }
        for eid in list_task_ids()
    ]
    rows.sort(key=lambda r: (r["suite"], r["env_id"]))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "env_id",
                "suite",
                "pattern",
                "reward_formula",
                "notes",
                "status",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
