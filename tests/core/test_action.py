import pytest

from rl_world_ascii.core.action import ActionSpec


def test_action_spec_n_equals_number_of_names():
    spec = ActionSpec(
        names=("NORTH", "SOUTH"),
        descriptions=("move up", "move down"),
    )
    assert spec.n == 2


def test_action_spec_index_of_valid_name():
    spec = ActionSpec(
        names=("NORTH", "SOUTH", "WAIT"),
        descriptions=("", "", ""),
    )
    assert spec.index_of("NORTH") == 0
    assert spec.index_of("WAIT") == 2


def test_action_spec_index_of_invalid_name_raises_key_error():
    spec = ActionSpec(names=("NORTH",), descriptions=("",))
    with pytest.raises(KeyError, match="FLY"):
        spec.index_of("FLY")


def test_action_spec_render_for_prompt_lists_every_action_with_description():
    spec = ActionSpec(
        names=("NORTH", "SOUTH", "WAIT"),
        descriptions=("move up", "move down", "do nothing"),
    )
    text = spec.render_for_prompt()
    assert "NORTH" in text
    assert "move up" in text
    assert "SOUTH" in text
    assert "WAIT" in text
    assert "do nothing" in text
    assert text.count("\n") >= 3  # header + 3 action lines


def test_action_spec_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        ActionSpec(names=("NORTH", "NORTH"), descriptions=("a", "b"))


def test_action_spec_names_and_descriptions_must_have_same_length():
    with pytest.raises(ValueError, match="same length"):
        ActionSpec(names=("NORTH", "SOUTH"), descriptions=("a",))


def test_action_spec_is_frozen():
    spec = ActionSpec(names=("NORTH",), descriptions=("a",))
    try:
        spec.names = ("FLY",)  # type: ignore[misc]
    except Exception as e:
        assert "frozen" in str(e).lower() or "can't set attribute" in str(e).lower() or "cannot assign" in str(e).lower()
    else:
        raise AssertionError("expected frozen dataclass")
