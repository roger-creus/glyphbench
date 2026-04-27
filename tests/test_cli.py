"""Tests for the glyphbench CLI."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from glyphbench import cli


def test_list_suites(capsys):
    rc = cli.main(["list-suites"])
    assert rc == 0
    out = capsys.readouterr().out.strip().split("\n")
    assert set(out) == {"minigrid", "minihack", "atari", "procgen", "craftax", "classics"}


def test_list_envs_filter_by_suite(capsys):
    rc = cli.main(["list-envs", "--suite", "atari"])
    assert rc == 0
    out = capsys.readouterr().out.strip().split("\n")
    assert all("/atari-" in e for e in out)
    assert len(out) > 10


def test_list_envs_excludes_dummy(capsys):
    cli.main(["list-envs"])
    out = capsys.readouterr().out
    assert "dummy" not in out


# NOTE: the `glyphbench eval` subcommand was removed — use `vf-eval glyphbench`
# (the canonical verifiers runner) directly. CLI now provides list-suites,
# list-envs, replay, and bundle only.


def test_bundle_creates_tarball_with_meta(tmp_path: Path):
    harness_dir = tmp_path / "Qwen_Qwen3-8B" / "history_cot"
    (harness_dir / "per_env").mkdir(parents=True)
    (harness_dir / "trajectories" / "glyphbench__atari-pong-v0").mkdir(parents=True)
    (harness_dir / "results.json").write_text(json.dumps({
        "model": "Qwen/Qwen3-8B",
        "n_envs": 1,
        "episodes_per_env": 25,
        "temperature": 0.7,
        "per_env": {"glyphbench/atari-pong-v0": {"mean_return": 0.0}},
    }))
    (harness_dir / "per_env" / "glyphbench__atari-pong-v0.json").write_text("{}")
    (harness_dir / "trajectories" / "glyphbench__atari-pong-v0" / "seed_1_ep_0.jsonl").write_text("{}\n")

    rc = cli.main(["bundle", str(harness_dir)])
    assert rc == 0

    tars = list(harness_dir.parent.glob("*.tar.gz"))
    assert len(tars) == 1, f"Expected exactly one tarball, got: {tars}"

    with tarfile.open(tars[0]) as tf:
        names = tf.getnames()
        assert any(n.endswith("results.json") for n in names)
        assert any(n.endswith("meta.json") for n in names)
        assert any("per_env" in n for n in names)
        assert any("trajectories" in n for n in names)

    meta = json.loads((harness_dir / "meta.json").read_text())
    assert meta["model"] == "Qwen/Qwen3-8B"
    assert meta["harness"] == "history_cot"
    assert meta["episodes_per_env"] == 25
    assert meta["protocol"]["max_turns"] == "env-native"


def test_bundle_rejects_dir_without_results_json(tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        cli._bundle_dir(empty, tar_output=None)


def test_bundle_custom_output_path(tmp_path: Path):
    harness_dir = tmp_path / "M" / "markov_cot"
    harness_dir.mkdir(parents=True)
    (harness_dir / "results.json").write_text(json.dumps({
        "model": "M", "n_envs": 0, "episodes_per_env": 1, "temperature": 0.7, "per_env": {},
    }))

    custom = tmp_path / "custom.tar.gz"
    rc = cli.main(["bundle", str(harness_dir), "--output", str(custom)])
    assert rc == 0
    assert custom.exists()


def test_build_turns_preserves_non_memory_grouping():
    rollout = {
        "prompt": [{"role": "user", "content": "u0"}],
        "completion": [
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
    }

    turns = cli._build_turns(rollout)

    assert turns == [
        {"user": "u0", "assistant": "a0", "memory": None},
        {"user": "u1", "assistant": "a1", "memory": None},
    ]


def test_build_turns_groups_memory_trajectory_steps():
    rollout_completion = [
        {
            "role": "assistant",
            "content": "<think>go</think><action>EAST</action>",
        },
        {"role": "user", "content": "[Memory Update]\n..."},
        {"role": "assistant", "content": "<think>update</think>moved east"},
    ]
    memory = {
        "previous_memory": "start seen",
        "stored_memory": "moved east",
        "action_response": [rollout_completion[0]],
        "memory_update_response": [rollout_completion[2]],
        "extraction_mode": "post_think",
    }
    rollout = {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "[Grid]\n@..\n..G"},
        ],
        "completion": [
            {"role": "assistant", "content": "<think>go</think><action>EAST</action>"},
            {"role": "user", "content": "[Memory Update]\n..."},
            {"role": "assistant", "content": "<think>update</think>moved east"},
        ],
        "trajectory": [
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "[Grid]\n@..\n..G"},
                ],
                "completion": rollout_completion,
                "extras": {"glyphbench_memory": memory},
            }
        ],
    }

    turns = cli._build_turns(rollout)

    assert len(turns) == 1
    assert turns[0]["user"] == "[Grid]\n@..\n..G"
    assert turns[0]["assistant"] == "<think>go</think><action>EAST</action>"
    assert turns[0]["memory"]["previous_memory"] == "start seen"
    assert turns[0]["memory"]["stored_memory"] == "moved east"


def test_build_turns_groups_memory_update_pairs_without_trajectory_extras():
    rollout = {
        "prompt": [
            {
                "role": "user",
                "content": "[Memory]\n<memory>\nstart seen\n</memory>\n\n[Grid]\n@..",
            }
        ],
        "completion": [
            {"role": "assistant", "content": "<action>EAST</action>"},
            {"role": "user", "content": "[Memory Update]\n..."},
            {"role": "assistant", "content": "<think>u</think><memory>moved east</memory>"},
            {"role": "user", "content": "[Memory]\n<memory>\nmoved east\n</memory>\n\n[Grid]\n.@."},
            {"role": "assistant", "content": "<action>EAST</action>"},
        ],
    }

    turns = cli._build_turns(rollout)

    assert len(turns) == 2
    assert turns[0]["user"].endswith("[Grid]\n@..")
    assert turns[0]["assistant"] == "<action>EAST</action>"
    assert turns[0]["memory"]["previous_memory"] == "start seen"
    assert turns[0]["memory"]["stored_memory"] == "moved east"
    assert turns[1]["user"].endswith("[Grid]\n.@.")
    assert turns[1]["assistant"] == "<action>EAST</action>"
    assert turns[1]["memory"] is None
