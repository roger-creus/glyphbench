from atlas_rl.core.observation import GridObservation
from atlas_rl.harness.prompt_builder import build_user_prompt
from atlas_rl.harness.state import EpisodeState, Subgoal


def _fresh_obs() -> GridObservation:
    return GridObservation(
        grid="@..",
        legend="@ — you\n. — floor",
        hud="Step: 5",
        message="",
    )


def test_turn_zero_cold_start_has_initial_plan_instruction():
    state = EpisodeState()
    obs = _fresh_obs()
    prompt = build_user_prompt(state, obs, turn_index=0)
    assert "FIRST TURN" in prompt
    assert "initial strategic plan" in prompt.lower()


def test_turn_n_prompt_shows_current_persistent_state():
    state = EpisodeState()
    state.strategic_plan = "win"
    state.tactical_plan = "go east"
    state.subgoals = [Subgoal("find key", done=False), Subgoal("open door", done=True)]
    state.lessons = ["kobolds take 2 hits"]
    state.recent_actions.append((3, "NORTH", "moved up"))
    state.recent_actions.append((4, "EAST", "bumped wall"))
    obs = _fresh_obs()
    prompt = build_user_prompt(state, obs, turn_index=5)
    assert "win" in prompt
    assert "go east" in prompt
    assert "find key" in prompt
    assert "open door" in prompt
    assert "kobolds take 2 hits" in prompt
    assert "turn 3" in prompt.lower() and "NORTH" in prompt
    assert "turn 4" in prompt.lower() and "EAST" in prompt
    # Current observation rendered
    assert "[Legend]" in prompt
    assert "[HUD]" in prompt
    assert "[Grid]" in prompt


def test_empty_subgoals_and_lessons_show_none_placeholder():
    state = EpisodeState()
    obs = _fresh_obs()
    prompt = build_user_prompt(state, obs, turn_index=1)
    assert "(none)" in prompt or "none" in prompt.lower()


def test_subgoal_done_marker():
    state = EpisodeState()
    state.subgoals = [Subgoal("a", done=True), Subgoal("b", done=False)]
    obs = _fresh_obs()
    prompt = build_user_prompt(state, obs, turn_index=1)
    assert "[x] a" in prompt
    assert "[ ] b" in prompt
