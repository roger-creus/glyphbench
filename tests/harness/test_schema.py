import pytest
from pydantic import ValidationError

from atlas_rl.harness.schema import HarnessOutput


def test_minimal_valid_output_only_action_required():
    out = HarnessOutput.model_validate({"action": "NORTH"})
    assert out.action == "NORTH"
    assert out.thinking == ""
    assert out.tactical_plan == ""
    assert out.strategic_plan_update is None
    assert out.lessons_to_add == []
    assert out.subgoals_update.add == []
    assert out.subgoals_update.mark_done == []


def test_action_required():
    with pytest.raises(ValidationError):
        HarnessOutput.model_validate({"thinking": "thoughts"})


def test_full_output_roundtrip():
    data = {
        "thinking": "I think the key is north",
        "strategic_plan_update": "get key then go to door",
        "tactical_plan": "move north twice",
        "subgoals_update": {"add": ["find key"], "mark_done": [0]},
        "lessons_to_add": ["doors require keys"],
        "action": "NORTH",
    }
    out = HarnessOutput.model_validate(data)
    assert out.strategic_plan_update == "get key then go to door"
    assert out.subgoals_update.add == ["find key"]
    assert out.subgoals_update.mark_done == [0]
    assert out.lessons_to_add == ["doors require keys"]
    assert out.action == "NORTH"


def test_strategic_plan_update_null_keeps_prior():
    out = HarnessOutput.model_validate({"action": "NORTH", "strategic_plan_update": None})
    assert out.strategic_plan_update is None


def test_strategic_plan_update_empty_string_replaces_with_empty():
    out = HarnessOutput.model_validate({"action": "NORTH", "strategic_plan_update": ""})
    assert out.strategic_plan_update == ""


def test_json_schema_export():
    schema = HarnessOutput.model_json_schema()
    assert "action" in schema["properties"]
    assert "action" in schema.get("required", [])
