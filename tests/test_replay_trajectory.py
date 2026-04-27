"""Tests for the standalone trajectory replay script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_replay_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_trajectory.py"
    spec = importlib.util.spec_from_file_location("replay_trajectory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_step_memory_reads_verifiers_extras():
    replay = _load_replay_module()

    memory = replay._step_memory(
        {"extras": {"glyphbench_memory": {"stored_memory": "door is east"}}}
    )

    assert memory == "door is east"


def test_replay_step_frame_includes_memory_block():
    replay = _load_replay_module()
    step = {
        "turn": 1,
        "action_name": "EAST",
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "terminated": False,
        "truncated": False,
        "parse_failed": False,
        "observation": "[Grid]\n@.",
        "extras": {"glyphbench_memory": {"stored_memory": "moved east once"}},
    }

    frame = replay._step_frame(step, Path("trajectory.jsonl"))

    assert "[Memory]" in frame
    assert "moved east once" in frame
