from atlas_rl.core.action import ActionSpec
from atlas_rl.harness.templating import render_system_prompt


def test_render_dummy_env_template_mentions_actions():
    action_spec = ActionSpec(
        names=("NORTH", "SOUTH", "EAST", "WEST", "NOOP"),
        descriptions=(
            "move up",
            "move down",
            "move right",
            "move left",
            "wait",
        ),
    )
    text = render_system_prompt(
        "__dummy-v0",
        env_name="Dummy",
        task_description="Reach the goal.",
        action_spec=action_spec,
        reward_description="+1 on goal",
        symbol_legend_summary="@ is you, G is goal",
        physics_notes="",
    )
    assert "NORTH" in text
    assert "move up" in text
    assert "Reach the goal" in text


def test_render_uses_base_response_format_block():
    action_spec = ActionSpec(names=("NOOP",), descriptions=("wait",))
    text = render_system_prompt(
        "__dummy-v0",
        env_name="Dummy",
        task_description="Do nothing.",
        action_spec=action_spec,
        reward_description="0",
        symbol_legend_summary="",
        physics_notes="",
    )
    # Every game inherits the JSON output format explanation from _base.j2
    assert "JSON" in text
    assert "action" in text
