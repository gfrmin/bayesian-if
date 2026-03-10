"""Tests for IF information-gathering tools."""

from bayesian_if.tools import (
    ExamineTool,
    IFTool,
    InventoryTool,
    LLMAdvisorTool,
    LookTool,
    _best_action_matching,
    _extract_keywords,
)
from bayesian_if.world import Observation
from tests.mock_world import MockWorld


def test_extract_keywords():
    text = "A rusty key sits on the old wooden table."
    kws = _extract_keywords(text)
    assert "rusty" in kws
    assert "key" in kws
    assert "table" in kws
    # Stop words excluded
    assert "the" not in kws
    assert "on" not in kws


def test_best_action_matching():
    actions = ["go north", "take key", "look", "wait"]
    assert _best_action_matching(actions, ["key", "rusty"]) == 1
    assert _best_action_matching(actions, ["north", "door"]) == 0
    assert _best_action_matching(actions, ["zzz"]) is None


def test_look_tool_returns_valid_index():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = world.valid_actions()

    tool = LookTool()
    result = tool.query(world, obs, actions)
    # Should return a valid index or None
    assert result is None or (0 <= result < len(actions))


def test_look_tool_does_not_consume_turn():
    world = MockWorld()
    world.reset()
    snapshot_before = world.save()
    obs = Observation(text="A room.", score=0, location="Start Room")
    actions = world.valid_actions()

    LookTool().query(world, obs, actions)

    # State should be unchanged
    actions_after = world.valid_actions()
    assert actions == actions_after


def test_examine_tool_returns_valid_index():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A rusty key sits on the Table.", score=0, location="Start Room")
    actions = world.valid_actions()

    tool = ExamineTool()
    result = tool.query(world, obs, actions)
    assert result is None or (0 <= result < len(actions))


def test_inventory_tool_returns_valid_index():
    world = MockWorld()
    world.reset()
    world.step("take key")
    obs = Observation(text="You have a key.", score=5, location="Start Room", inventory=("key",))
    actions = world.valid_actions()

    tool = InventoryTool()
    result = tool.query(world, obs, actions)
    assert result is None or (0 <= result < len(actions))


def test_inventory_tool_does_not_consume_turn():
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = world.valid_actions()
    snapshot = world.save()

    InventoryTool().query(world, obs, actions)

    world_actions = world.valid_actions()
    assert actions == world_actions


def test_llm_advisor_with_mock():
    """LLM advisor with a mock generate function."""
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room with a key.", score=0)
    actions = world.valid_actions()

    # Mock LLM that always returns "1"
    tool = LLMAdvisorTool(generate_fn=lambda _prompt: "1")
    result = tool.query(world, obs, actions)
    assert result == 1


def test_llm_advisor_handles_bad_response():
    """LLM advisor handles non-numeric responses gracefully."""
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = world.valid_actions()

    tool = LLMAdvisorTool(generate_fn=lambda _prompt: "I think you should explore")
    result = tool.query(world, obs, actions)
    assert result is None


def test_llm_advisor_handles_out_of_range():
    """LLM advisor handles out-of-range indices."""
    world = MockWorld()
    world.reset()
    obs = Observation(text="A room.", score=0)
    actions = world.valid_actions()

    tool = LLMAdvisorTool(generate_fn=lambda _prompt: "999")
    result = tool.query(world, obs, actions)
    assert result is None


def test_tool_config_conversion():
    """Tools convert to credence ToolConfig correctly."""
    tool = LookTool()
    config = tool.to_tool_config()
    assert config.cost == 0.0
    assert len(config.coverage_by_category) == 5
    assert all(0.0 <= c <= 1.0 for c in config.coverage_by_category)
