"""End-to-end rollout test against a canned-response mock client."""

from __future__ import annotations

import pytest

import glyphbench
from glyphbench.core import make_env


@pytest.mark.asyncio
async def test_mock_rollout_accumulates_reward():
    """Drive the env through a scripted sequence of actions and verify the
    rubric sees a positive episodic return when the dummy env reaches its
    goal (east/south east sequence navigates 0,0 → 2,2 in 4 steps)."""

    env = glyphbench.load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        n_frames=4,
        max_turns=10,
    )
    # Scripted assistant replies — 4 moves to reach the goal.
    scripted = [
        "<think>right</think><action>EAST</action>",
        "<think>down</think><action>SOUTH</action>",
        "<think>right</think><action>EAST</action>",
        "<think>down</think><action>SOUTH</action>",
    ]

    # Drive setup_state + four env_response cycles by hand (mocking the model).
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)

    for reply in scripted:
        state["trajectory"].append({"reward": None, "extras": {}})
        await env.env_response(
            [{"role": "assistant", "content": reply}], state
        )
        if state["done"]:
            break

    assert state["episode_return"] == pytest.approx(1.0)
    assert state["terminated"] is True
    assert state["truncated"] is False
    assert state["parse_failures"] == 0


@pytest.mark.asyncio
async def test_mock_rollout_parse_failures_accumulate():
    env = glyphbench.load_environment(
        task_id="glyphbench/__dummy-v0",
        num_episodes=1,
        n_frames=4,
        max_turns=3,
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    for _ in range(3):
        state["trajectory"].append({"reward": None, "extras": {}})
        await env.env_response(
            [{"role": "assistant", "content": "garbled output with no action"}],
            state,
        )
        if state["done"]:
            break
    assert state["parse_failures"] == 3
    assert state["truncated"] is True
    assert state["terminated"] is False
