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


def test_eval_parser_defaults():
    parser = cli._build_parser()
    args = parser.parse_args(["eval", "Qwen/Qwen3-8B"])
    assert args.model == "Qwen/Qwen3-8B"
    assert args.harness == "history_cot"
    assert args.episodes == 25
    assert args.suite is None
    assert args.env is None


def test_eval_parser_scope_mutex():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "M", "--suite", "atari", "--env", "x"])


def test_eval_parser_harness_all_accepted():
    parser = cli._build_parser()
    args = parser.parse_args(["eval", "M", "--harness", "all"])
    assert args.harness == "all"


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
