"""load_environment exposes the filter kwargs from list_task_ids."""
from __future__ import annotations

import pytest

import glyphbench  # noqa: F401 — registers all envs
from glyphbench.verifiers_integration.env import _resolve_env_ids


def test_resolve_no_filter_returns_all_minus_dummy():
    ids = _resolve_env_ids(None, None, None, None, None)
    assert all("__dummy" not in i for i in ids)
    assert any(i.startswith("glyphbench/atari-") for i in ids)


def test_resolve_include_suites():
    ids = _resolve_env_ids(None, ["minigrid"], None, None, None)
    assert all(i.startswith("glyphbench/minigrid-") for i in ids)


def test_resolve_exclude_suites():
    ids = _resolve_env_ids(None, None, ["atari"], None, None)
    assert all(not i.startswith("glyphbench/atari-") for i in ids)


def test_resolve_explicit_task_id_takes_precedence():
    ids = _resolve_env_ids("glyphbench/minigrid-empty-5x5-v0", ["atari"], None, None, None)
    assert ids == ["glyphbench/minigrid-empty-5x5-v0"]


def test_resolve_include_pattern_via_tasks():
    ids = _resolve_env_ids(None, None, None, ["glyphbench/minigrid-*-v0"], None)
    assert all(i.startswith("glyphbench/minigrid-") for i in ids)


def test_resolve_exclude_pattern():
    ids = _resolve_env_ids(None, None, None, None, ["glyphbench/atari-*-v0"])
    assert all(not i.startswith("glyphbench/atari-") for i in ids)
