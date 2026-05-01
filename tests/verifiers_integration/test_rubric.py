"""Tests for EpisodicReturnRubric."""

from __future__ import annotations

import pytest

from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric


@pytest.fixture
def rubric():
    return EpisodicReturnRubric(parser=GlyphbenchXMLParser())


async def _call(fn, **kw):
    return await fn(**kw)


@pytest.mark.asyncio
async def test_episodic_return_sums_per_step_rewards(rubric):
    state = {
        "episode_return": 1.5,
        "trajectory": [{"reward": 0.5}, {"reward": 0.5}, {"reward": 0.5}],
        "parse_failures": 0,
        "terminated": True,
        "truncated": False,
    }
    r = await rubric.episodic_return(state=state)
    assert r == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_forfeit_rate_via_trajectory(rubric):
    state = {
        "episode_return": 0.0,
        "num_action_turns": 4,
        "forfeit_count": 1,
        "terminated": False,
        "truncated": True,
    }
    r = await rubric.forfeit_rate(state=state)
    assert r == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_forfeit_rate_empty_trajectory_returns_zero_via_fixture(rubric):
    state = {"episode_return": 0.0, "num_action_turns": 0, "forfeit_count": 0}
    r = await rubric.forfeit_rate(state=state)
    assert r == 0.0


@pytest.mark.asyncio
async def test_episode_length(rubric):
    state = {"trajectory": [{}] * 7}
    assert await rubric.episode_length(state=state) == 7.0


@pytest.mark.asyncio
async def test_terminated_and_truncated_flags(rubric):
    assert await rubric.episode_terminated_rate(state={"terminated": True}) == 1.0
    assert await rubric.episode_terminated_rate(state={"terminated": False}) == 0.0
    assert await rubric.episode_truncated_max_turns_rate(state={"truncated": True}) == 1.0
    assert await rubric.episode_truncated_max_turns_rate(state={"truncated": False}) == 0.0


def test_episode_terminated_rate_renamed():
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    metric_names = {m.__name__ for m in rubric.funcs}
    assert "episode_terminated_rate" in metric_names
    assert "terminated_flag" not in metric_names


def test_episode_truncated_max_turns_rate_renamed():
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    metric_names = {m.__name__ for m in rubric.funcs}
    assert "episode_truncated_max_turns_rate" in metric_names
    assert "truncated_flag" not in metric_names


def test_forfeit_rate_replaces_parse_failure_rate():
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    metric_names = {m.__name__ for m in rubric.funcs}
    assert "forfeit_rate" in metric_names
    assert "parse_failure_rate" not in metric_names


def test_action_completion_truncation_rate_metric_present():
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    metric_names = {m.__name__ for m in rubric.funcs}
    assert "action_completion_truncation_rate" in metric_names


def test_memory_completion_truncation_rate_metric_present():
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    metric_names = {m.__name__ for m in rubric.funcs}
    assert "memory_completion_truncation_rate" in metric_names


def test_memory_parse_failure_rate_metric_present():
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    metric_names = {m.__name__ for m in rubric.funcs}
    assert "memory_parse_failure_rate" in metric_names


def test_forfeit_rate_value():
    import asyncio
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    state = {"num_action_turns": 4, "forfeit_count": 1}
    rate = asyncio.run(rubric.forfeit_rate(state=state))
    assert rate == pytest.approx(0.25)


def test_forfeit_rate_zero_turns_returns_zero():
    import asyncio
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    state = {"num_action_turns": 0, "forfeit_count": 0}
    rate = asyncio.run(rubric.forfeit_rate(state=state))
    assert rate == 0.0


def test_action_completion_truncation_rate_value():
    import asyncio
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    state = {"num_action_turns": 5, "action_completion_truncations": 2}
    rate = asyncio.run(rubric.action_completion_truncation_rate(state=state))
    assert rate == pytest.approx(0.4)


def test_memory_completion_truncation_rate_value():
    import asyncio
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    state = {"num_memory_turns": 4, "memory_completion_truncations": 1}
    rate = asyncio.run(rubric.memory_completion_truncation_rate(state=state))
    assert rate == pytest.approx(0.25)


def test_memory_parse_failure_rate_value():
    import asyncio
    rubric = EpisodicReturnRubric(parser=GlyphbenchXMLParser())
    state = {"num_memory_turns": 4, "memory_parse_failures": 3}
    rate = asyncio.run(rubric.memory_parse_failure_rate(state=state))
    assert rate == pytest.approx(0.75)
