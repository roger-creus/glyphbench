"""Per-run output directory manager: parquet summaries + optional JSONL trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rl_world_ascii.core.metrics import TurnMetrics


@dataclass
class EpisodeRecord:
    env_id: str
    seed: int
    episode_idx: int
    episode_return: float
    episode_length: int
    terminated_reason: str
    turn_metrics: list[TurnMetrics]
    extras: dict[str, Any] = field(default_factory=dict)


def _slugify_env_id(env_id: str) -> str:
    slug = env_id.replace("/", "__").replace(":", "_")
    # Collapse runs of underscores (e.g. "rl_world_ascii/__dummy-v0" -> "rl_world_ascii__dummy-v0").
    while "___" in slug:
        slug = slug.replace("___", "__")
    return slug


class RunStorage:
    def __init__(
        self,
        *,
        base_dir: Path | str,
        run_id: str,
        trajectory_logging: bool = False,
    ) -> None:
        self._run_dir = Path(base_dir) / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_logging = trajectory_logging
        self._episodes: list[EpisodeRecord] = []

    def record_episode(self, episode: EpisodeRecord) -> None:
        self._episodes.append(episode)
        if self._trajectory_logging:
            self._write_trajectory(episode)

    def _write_trajectory(self, episode: EpisodeRecord) -> None:
        traj_path = (
            self._run_dir
            / "trajectories"
            / _slugify_env_id(episode.env_id)
            / f"seed_{episode.seed}"
            / f"episode_{episode.episode_idx}.jsonl"
        )
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        records = episode.extras.get("trajectory", [])
        with traj_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def finalize(self) -> None:
        self._write_turns_parquet()
        self._write_summary_parquet()

    def _write_turns_parquet(self) -> None:
        rows: list[dict[str, Any]] = []
        for ep in self._episodes:
            for m in ep.turn_metrics:
                row = m.to_dict()
                row["env_id"] = ep.env_id
                row["seed"] = ep.seed
                row["episode_idx"] = ep.episode_idx
                rows.append(row)
        if not rows:
            return
        df = pd.DataFrame(rows)
        out = self._run_dir / "turns.parquet"
        pq.write_table(pa.Table.from_pandas(df), out)

    def _write_summary_parquet(self) -> None:
        rows: list[dict[str, Any]] = []
        for ep in self._episodes:
            total_in = sum(m.tokens_in for m in ep.turn_metrics)
            total_out = sum(m.tokens_out for m in ep.turn_metrics)
            total_reasoning = sum(m.tokens_reasoning for m in ep.turn_metrics)
            total_cost = sum(m.dollar_cost_turn for m in ep.turn_metrics)
            total_wall = sum(m.wall_time_s for m in ep.turn_metrics)
            parse_failures = sum(1 for m in ep.turn_metrics if m.action_parse_error)
            parse_failure_rate = (
                parse_failures / len(ep.turn_metrics) if ep.turn_metrics else 0.0
            )
            latencies = sorted(m.latency_provider_s for m in ep.turn_metrics)
            mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
            p95_lat = latencies[int(0.95 * len(latencies))] if latencies else 0.0
            rows.append(
                {
                    "env_id": ep.env_id,
                    "seed": ep.seed,
                    "episode_idx": ep.episode_idx,
                    "episode_return": ep.episode_return,
                    "episode_length": ep.episode_length,
                    "terminated_reason": ep.terminated_reason,
                    "total_tokens_in": total_in,
                    "total_tokens_out": total_out,
                    "total_tokens_reasoning": total_reasoning,
                    "total_dollar_cost": total_cost,
                    "total_wall_time_s": total_wall,
                    "mean_latency_s": mean_lat,
                    "p95_latency_s": p95_lat,
                    "action_parse_failure_rate": parse_failure_rate,
                }
            )
        if not rows:
            return
        df = pd.DataFrame(rows)
        out = self._run_dir / "summary.parquet"
        pq.write_table(pa.Table.from_pandas(df), out)

    @property
    def run_dir(self) -> Path:
        return self._run_dir
