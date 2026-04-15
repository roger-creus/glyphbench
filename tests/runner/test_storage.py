import json

import pandas as pd

from rl_world_ascii.core.metrics import TurnMetrics
from rl_world_ascii.runner.storage import EpisodeRecord, RunStorage


def _metric(i: int) -> TurnMetrics:
    return TurnMetrics(
        turn_index=i, wall_time_s=0.01, reward=0.0,
        terminated=False, truncated=False,
        action_index=0, action_name="NORTH",
        action_parse_error=False, action_parse_retries=0, action_fell_back_to_noop=False,
        tokens_in=100, tokens_out=20, tokens_reasoning=0,
        latency_provider_s=0.05, dollar_cost_turn=0.001,
        subgoals_added=0, subgoals_marked_done=0, lessons_added=0,
        tactical_plan_changed=False, strategic_plan_changed=False,
        prompt_char_count=500, prompt_token_count=120,
    )


def test_storage_writes_turns_parquet(tmp_path):
    storage = RunStorage(base_dir=tmp_path, run_id="unit-test")
    episode = EpisodeRecord(
        env_id="rl_world_ascii/__dummy-v0",
        seed=0,
        episode_idx=0,
        episode_return=1.0,
        episode_length=3,
        terminated_reason="success",
        turn_metrics=[_metric(0), _metric(1), _metric(2)],
        extras={},
    )
    storage.record_episode(episode)
    storage.finalize()
    turns_path = tmp_path / "unit-test" / "turns.parquet"
    assert turns_path.exists()
    df = pd.read_parquet(turns_path)
    assert len(df) == 3
    assert set(df["action_name"].unique()) == {"NORTH"}


def test_storage_writes_summary_parquet(tmp_path):
    storage = RunStorage(base_dir=tmp_path, run_id="unit-test-2")
    for seed in [0, 1]:
        storage.record_episode(
            EpisodeRecord(
                env_id="rl_world_ascii/__dummy-v0",
                seed=seed,
                episode_idx=0,
                episode_return=float(seed),
                episode_length=5,
                terminated_reason="success",
                turn_metrics=[_metric(0)],
                extras={},
            )
        )
    storage.finalize()
    summary_path = tmp_path / "unit-test-2" / "summary.parquet"
    df = pd.read_parquet(summary_path)
    assert len(df) == 2
    assert set(df["seed"].tolist()) == {0, 1}
    assert "total_dollar_cost" in df.columns


def test_storage_writes_jsonl_trajectory_when_enabled(tmp_path):
    storage = RunStorage(
        base_dir=tmp_path, run_id="traj-test", trajectory_logging=True
    )
    storage.record_episode(
        EpisodeRecord(
            env_id="rl_world_ascii/__dummy-v0",
            seed=0,
            episode_idx=0,
            episode_return=1.0,
            episode_length=1,
            terminated_reason="success",
            turn_metrics=[_metric(0)],
            extras={"trajectory": [{"turn": 0, "prompt": "...", "response": "..."}]},
        )
    )
    storage.finalize()
    traj_path = (
        tmp_path
        / "traj-test"
        / "trajectories"
        / "rl_world_ascii__dummy-v0"
        / "seed_0"
        / "episode_0.jsonl"
    )
    assert traj_path.exists()
    lines = traj_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["turn"] == 0
