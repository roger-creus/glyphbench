from rl_world_ascii.harness.state import EpisodeState, Subgoal


def test_fresh_state_has_empty_fields():
    s = EpisodeState()
    assert s.strategic_plan == ""
    assert s.tactical_plan == ""
    assert s.subgoals == []
    assert s.lessons == []
    assert len(s.recent_actions) == 0


def test_recent_actions_max_len_five():
    s = EpisodeState()
    for i in range(10):
        s.recent_actions.append((i, f"ACTION_{i}", f"outcome {i}"))
    assert len(s.recent_actions) == 5
    # Oldest two dropped, we kept turns 5..9
    turns = [t for (t, _, _) in s.recent_actions]
    assert turns == [5, 6, 7, 8, 9]


def test_subgoal_fields():
    sg = Subgoal(text="find key")
    assert sg.text == "find key"
    assert sg.done is False


def test_reset_clears_everything():
    s = EpisodeState()
    s.strategic_plan = "win"
    s.tactical_plan = "go north"
    s.subgoals.append(Subgoal(text="find key"))
    s.lessons.append("kobolds take 2 hits")
    s.recent_actions.append((0, "MOVE_N", "ok"))
    s.reset()
    assert s.strategic_plan == ""
    assert s.tactical_plan == ""
    assert s.subgoals == []
    assert s.lessons == []
    assert len(s.recent_actions) == 0
