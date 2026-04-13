from rl_world_ascii.core.metrics import TurnMetrics


def test_turn_metrics_required_fields():
    m = TurnMetrics(
        turn_index=5,
        wall_time_s=0.123,
        reward=1.0,
        terminated=False,
        truncated=False,
        action_index=2,
        action_name="EAST",
        action_parse_error=False,
        action_parse_retries=0,
        action_fell_back_to_noop=False,
        tokens_in=1234,
        tokens_out=56,
        tokens_reasoning=0,
        latency_provider_s=0.45,
        dollar_cost_turn=0.0,
        subgoals_added=0,
        subgoals_marked_done=0,
        lessons_added=0,
        tactical_plan_changed=False,
        strategic_plan_changed=False,
        prompt_char_count=9000,
        prompt_token_count=2300,
    )
    assert m.turn_index == 5
    assert m.reward == 1.0
    assert m.action_name == "EAST"


def test_turn_metrics_to_dict_has_all_fields_as_keys():
    m = TurnMetrics(
        turn_index=0, wall_time_s=0.0, reward=0.0,
        terminated=False, truncated=False,
        action_index=0, action_name="NOOP",
        action_parse_error=False, action_parse_retries=0, action_fell_back_to_noop=False,
        tokens_in=0, tokens_out=0, tokens_reasoning=0,
        latency_provider_s=0.0, dollar_cost_turn=0.0,
        subgoals_added=0, subgoals_marked_done=0, lessons_added=0,
        tactical_plan_changed=False, strategic_plan_changed=False,
        prompt_char_count=0, prompt_token_count=0,
    )
    d = m.to_dict()
    for k in (
        "turn_index", "wall_time_s", "reward", "terminated", "truncated",
        "action_index", "action_name", "action_parse_error", "action_parse_retries",
        "action_fell_back_to_noop", "tokens_in", "tokens_out", "tokens_reasoning",
        "latency_provider_s", "dollar_cost_turn", "subgoals_added",
        "subgoals_marked_done", "lessons_added", "tactical_plan_changed",
        "strategic_plan_changed", "prompt_char_count", "prompt_token_count",
    ):
        assert k in d
