"""Tests for GlyphbenchMultiTurnEnv + load_environment."""

from __future__ import annotations

import json

import pytest

import glyphbench.envs.dummy  # ensure dummy registered
from glyphbench.verifiers_integration import GlyphbenchMultiTurnEnv, load_environment


def test_load_environment_returns_multi_turn_env():
    env = load_environment(
        env_id="glyphbench/__dummy-v0",
        num_episodes=2,
        n_frames=4,
        max_output_tokens=512,
    )
    assert isinstance(env, GlyphbenchMultiTurnEnv)


def test_load_environment_dataset_shape():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=3)
    assert len(env.dataset) == 3
    for row in env.dataset:
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        assert info["env_id"] == "glyphbench/__dummy-v0"
        assert isinstance(info["seed"], int)
    # Seeds should be distinct.
    seeds = [
        json.loads(r["info"])["seed"] if isinstance(r["info"], str) else r["info"]["seed"]
        for r in env.dataset
    ]
    assert len(set(seeds)) == len(seeds)


def test_load_environment_multiple_env_ids():
    env = load_environment(
        env_id=["glyphbench/__dummy-v0"],
        num_episodes=2,
    )
    assert len(env.dataset) == 2


def test_load_environment_rejects_unknown_id():
    with pytest.raises(KeyError):
        load_environment(env_id="glyphbench/does-not-exist-v0", num_episodes=1)


@pytest.mark.asyncio
async def test_setup_state_creates_game_and_initial_obs():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=1)
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    assert "game" in state
    assert state["current_obs"]  # non-empty
    assert state["parse_failures"] == 0
    assert state["episode_return"] == 0.0
    assert state["done"] is False
    assert state["terminated"] is False
    assert state["truncated"] is False
    # The prompt must have been populated with system + initial user turn.
    assert len(state["prompt"]) == 2
    assert state["prompt"][0]["role"] == "system"
    assert state["prompt"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_env_response_applies_action_and_updates_state():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=1)
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [{"reward": None, "extras": {}}],  # verifiers appends before env_response
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    state["trajectory"] = [{"reward": None, "extras": {}}]
    model_reply = "<think>go east</think><action>EAST</action>"
    messages = [{"role": "assistant", "content": model_reply}]
    response = await env.env_response(messages, state)
    # Returns a list of 1 user message with the next observation.
    assert isinstance(response, list) and len(response) == 1
    assert response[0]["role"] == "user"
    assert state["trajectory"][-1]["reward"] is not None
    # One frame accumulated for the action just taken.
    assert len(state["frames"]) == 1


@pytest.mark.asyncio
async def test_is_done_terminates_on_game_end():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=1)
    state: dict = {"done": False}
    assert await env.is_done(state) is False
    state["done"] = True
    assert await env.is_done(state) is True
