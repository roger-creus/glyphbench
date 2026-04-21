"""Test that history-mode prompts in eval/run_eval.py preserve the correct
causal chain: history[t] pairs the obs the model saw at turn t with the
action chosen at turn t and the reward received for that action.

Regression test for the inverted-causal-chain bug where the rendered
history made pre-action observations look like post-action states.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

import glyphbench  # noqa: F401 — register envs

# Load eval/run_eval.py as a module (it lives outside the src/ package tree).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_EVAL_PATH = _REPO_ROOT / "eval" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", _RUN_EVAL_PATH)
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = run_eval
_spec.loader.exec_module(run_eval)


class RecordingEngine:
    """Duck-typed replacement for VLLMEngine that records prompts and replays
    scripted responses in order. Each call to generate_batch returns the next
    chunk_n responses from the queue and appends the messages_list to history.
    """

    def __init__(self, scripted_actions: list[str]) -> None:
        # scripted_actions is a list of action names to return, one per call-item.
        # We serve them in order across all generate_batch calls.
        self._queue: list[str] = list(scripted_actions)
        self.prompts_per_turn: list[list[list[dict[str, str]]]] = []

    def generate_batch(
        self, messages_list: list[list[dict[str, str]]]
    ) -> list[dict[str, Any]]:
        # Deep-copy-ish: store the list of messages each turn
        self.prompts_per_turn.append([list(msgs) for msgs in messages_list])
        n = len(messages_list)
        if len(self._queue) < n:
            raise RuntimeError("scripted queue exhausted")
        actions = self._queue[:n]
        self._queue = self._queue[n:]
        return [
            {
                "text": '{"action": "' + a + '"}',
                "input_tokens": 10,
                "output_tokens": 5,
                "latency_s": 0.0,
            }
            for a in actions
        ]


def _grid_from_user_msg(user_content: str) -> str:
    """Extract the `[Grid]` block from a rendered observation string."""
    lines = user_content.split("\n")
    out: list[str] = []
    in_grid = False
    for line in lines:
        if line == "[Grid]":
            in_grid = True
            continue
        if in_grid and line.startswith("[") and line.endswith("]"):
            break
        if in_grid:
            out.append(line)
    # Strip trailing empties
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _extract_recent_history_block(user_content: str) -> str:
    """Return substring between '[Recent History]' and the next top-level
    bracketed header (e.g. '[Observation'). Returns empty string if no block.
    """
    marker = "[Recent History]"
    if marker not in user_content:
        return ""
    start = user_content.index(marker) + len(marker)
    rest = user_content[start:]
    # Next header starts with "\n[" on its own line
    # Find "[Observation" or similar.
    next_header_idx = rest.find("\n[Observation")
    if next_header_idx == -1:
        return rest.strip()
    return rest[:next_header_idx].strip()


def test_history_mode_pairs_obs_with_action_chosen_from_that_obs() -> None:
    """In history mode, history[t] must pair (obs_t_as_seen, action_t_chosen,
    reward_t). The rendered history must present obs BEFORE action/reward so
    the causal chain (observe -> act -> receive reward) is unambiguous."""
    # Scripted action sequence: 4 distinct moves + noop, so on turn 5 the
    # model has 4 entries of history (history_len=5, last 5 kept, only 4 exist).
    actions = ["EAST", "EAST", "SOUTH", "SOUTH", "NOOP"]
    engine = RecordingEngine(scripted_actions=actions)

    mode = run_eval.HARNESS_MODES["history_zeroshot"]
    results = run_eval.run_batched_episodes(
        engine=engine,
        env_id="glyphbench/__dummy-v0",
        seeds=[0],
        max_turns=5,
        batch_size=1,
        mode=mode,
    )

    # Trajectory sanity: 4 steps before goal is reached on the 4th action.
    # The 4th SOUTH reaches (2,2) so episode terminates at turn 4.
    ep = results[0]
    assert ep.episode_length == 4, (
        f"expected 4 steps until goal, got {ep.episode_length}"
    )

    # We want to inspect the prompts on turn 4 (the last turn before goal),
    # which has 3 history entries for turns 1, 2, 3.
    assert len(engine.prompts_per_turn) >= 4, (
        f"expected at least 4 batched calls, got {len(engine.prompts_per_turn)}"
    )
    turn4_msgs = engine.prompts_per_turn[3][0]  # batch 0, episode 0
    user_content = turn4_msgs[1]["content"]

    # -- Assertion 1: history block exists and has 3 entries --
    hist_block = _extract_recent_history_block(user_content)
    assert hist_block, "expected [Recent History] section in history_zeroshot mode"
    assert "Step 1" in hist_block
    assert "Step 2" in hist_block
    assert "Step 3" in hist_block

    # -- Assertion 2: each entry shows an obs AND the action chosen FROM that obs --
    # The dummy env's obs includes position in the grid. Build the expected
    # pre-obs for each turn independently:
    from glyphbench.envs.dummy.env import DummyEnv
    ref_env = DummyEnv(max_turns=5)
    ref_env.reset(seed=0)

    expected_pre_obs: list[str] = []
    # Turn 1: obs is post-reset (pos 0,0)
    expected_pre_obs.append(_grid_section(ref_env._render_current_observation().render()))
    ref_env.step(ref_env.action_spec.index_of("EAST"))
    # Turn 2: pos (1,0)
    expected_pre_obs.append(_grid_section(ref_env._render_current_observation().render()))
    ref_env.step(ref_env.action_spec.index_of("EAST"))
    # Turn 3: pos (2,0)
    expected_pre_obs.append(_grid_section(ref_env._render_current_observation().render()))
    # (ref_env not stepped further — turn 4's obs is what the model sees "now")

    # Each history entry should contain the grid that was SHOWN TO THE MODEL
    # at that turn.
    for i in range(3):
        step_idx = i + 1
        step_header = f"[Step {step_idx}]"
        assert step_header in hist_block, f"missing {step_header} in history"
        # Slice out this step's block
        start = hist_block.index(step_header)
        end = (
            hist_block.index(f"[Step {step_idx + 1}]")
            if f"[Step {step_idx + 1}]" in hist_block
            else len(hist_block)
        )
        step_block = hist_block[start:end]

        # Grid from pre-obs must appear in step block
        assert expected_pre_obs[i] in step_block, (
            f"Step {step_idx}: expected pre-action grid\n{expected_pre_obs[i]}\n"
            f"not found in step block:\n{step_block}"
        )
        # Action actually chosen at that turn must appear (tolerant to
        # label wording, just requires the action name is present).
        assert actions[i] in step_block, (
            f"Step {step_idx}: expected action {actions[i]} in\n{step_block}"
        )

    # -- Assertion 3: causal ordering — grid must appear BEFORE 'Action:' in each
    # entry so the reader sees "I observed X, then I chose Y, then I got R".
    # This prevents the inverted-chain misreading where a pre-action grid looks
    # like a post-action resulting state.
    for i in range(3):
        step_idx = i + 1
        step_header = f"[Step {step_idx}]"
        start = hist_block.index(step_header)
        end = (
            hist_block.index(f"[Step {step_idx + 1}]")
            if f"[Step {step_idx + 1}]" in hist_block
            else len(hist_block)
        )
        step_block = hist_block[start:end]
        # The grid (pre-action observation) must appear BEFORE the action name
        # in each step block so the reader parses the entry in temporal
        # order: "I observed X, then chose action Y, then got reward R."
        grid = expected_pre_obs[i]
        grid_first_line = grid.split("\n")[0]
        grid_pos = step_block.find(grid_first_line)
        action_pos = step_block.find(actions[i])
        assert grid_pos != -1, f"Step {step_idx}: grid missing"
        assert action_pos != -1, f"Step {step_idx}: action name missing"
        assert grid_pos < action_pos, (
            f"Step {step_idx}: causal chain inverted — grid at pos {grid_pos} "
            f"appears AFTER the action name at pos {action_pos}. "
            f"History must present obs BEFORE action/reward so readers parse "
            f"'observed X -> chose Y -> got R', not 'chose Y -> got R -> saw X'."
        )


def test_markov_mode_shows_no_history() -> None:
    """Both markov_zeroshot and markov_cot must NEVER include a [Recent History]
    section. Markov = current obs only."""
    for mode_name in ("markov_zeroshot", "markov_cot"):
        actions = ["EAST", "EAST", "SOUTH", "SOUTH"]
        engine = RecordingEngine(scripted_actions=actions)
        mode = run_eval.HARNESS_MODES[mode_name]
        run_eval.run_batched_episodes(
            engine=engine,
            env_id="glyphbench/__dummy-v0",
            seeds=[0],
            max_turns=5,
            batch_size=1,
            mode=mode,
        )
        # Check prompts across ALL turns — none should have a history section.
        for turn_idx, batch_msgs in enumerate(engine.prompts_per_turn):
            for msgs in batch_msgs:
                user_content = msgs[1]["content"]
                assert "[Recent History]" not in user_content, (
                    f"mode={mode_name}: turn {turn_idx} prompt contains "
                    f"[Recent History] — markov modes must not render history"
                )


def test_history_mode_first_turn_has_empty_or_no_history() -> None:
    """On turn 1 there are no past steps. The prompt should either omit the
    [Recent History] block entirely or show '(no history yet)'."""
    engine = RecordingEngine(scripted_actions=["EAST", "EAST", "SOUTH", "SOUTH"])
    mode = run_eval.HARNESS_MODES["history_zeroshot"]
    run_eval.run_batched_episodes(
        engine=engine,
        env_id="glyphbench/__dummy-v0",
        seeds=[0],
        max_turns=5,
        batch_size=1,
        mode=mode,
    )
    turn1 = engine.prompts_per_turn[0][0]
    user_content = turn1[1]["content"]
    # No Step 1 entry should exist in a history block on turn 1
    hist_block = _extract_recent_history_block(user_content)
    if hist_block:
        # If present, must not contain any step entries
        assert "[Step 1]" not in hist_block, (
            f"turn 1 history block should not contain step entries, got:\n{hist_block}"
        )


def test_history_add_stores_pre_action_obs() -> None:
    """Direct unit test on StepHistory.add: verify stored tuple is
    (pre_action_obs, action_name, reward_for_that_action)."""
    hist = run_eval.StepHistory(max_len=3)
    hist.add("OBS_A", "EAST", 0.25)
    hist.add("OBS_B", "SOUTH", -0.10)
    assert len(hist.entries) == 2
    assert hist.entries[0] == ("OBS_A", "EAST", 0.25)
    assert hist.entries[1] == ("OBS_B", "SOUTH", -0.10)


def _grid_section(rendered_obs: str) -> str:
    """Extract everything under `[Grid]\\n` up to the next `[...]` section
    header or end of string. Strips trailing blank lines."""
    lines = rendered_obs.split("\n")
    out: list[str] = []
    in_grid = False
    for line in lines:
        if line == "[Grid]":
            in_grid = True
            continue
        if in_grid and line.startswith("[") and line.endswith("]"):
            break
        if in_grid:
            out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)
