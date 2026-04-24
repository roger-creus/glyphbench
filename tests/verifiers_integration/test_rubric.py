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
async def test_parse_failure_rate(rubric):
    state = {
        "episode_return": 0.0,
        "trajectory": [{}, {}, {}, {}],
        "parse_failures": 1,
        "terminated": False,
        "truncated": True,
    }
    r = await rubric.parse_failure_rate(state=state)
    assert r == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_parse_failure_rate_empty_trajectory_returns_zero(rubric):
    state = {"episode_return": 0.0, "trajectory": [], "parse_failures": 0}
    r = await rubric.parse_failure_rate(state=state)
    assert r == 0.0


@pytest.mark.asyncio
async def test_episode_length(rubric):
    state = {"trajectory": [{}] * 7}
    assert await rubric.episode_length(state=state) == 7.0


@pytest.mark.asyncio
async def test_terminated_and_truncated_flags(rubric):
    assert await rubric.terminated_flag(state={"terminated": True}) == 1.0
    assert await rubric.terminated_flag(state={"terminated": False}) == 0.0
    assert await rubric.truncated_flag(state={"truncated": True}) == 1.0
    assert await rubric.truncated_flag(state={"truncated": False}) == 0.0
