"""Tests for the IFAgent — full integration with mock world."""

import numpy as np

from bayesian_if.agent import IFAgent
from bayesian_if.tools import DEFAULT_TOOLS, LLMAdvisorTool
from tests.mock_world import MockWorld


def test_agent_plays_mock_world():
    """Agent should be able to play the mock world without crashing."""
    world = MockWorld()
    agent = IFAgent(world=world, tools=list(DEFAULT_TOOLS))
    result = agent.play_game(max_steps=50)

    assert result.steps_taken > 0
    assert result.final_score >= 0
    assert len(result.steps) == result.steps_taken


def test_agent_with_llm_mock():
    """Agent with a mock LLM advisor should work end-to-end."""
    world = MockWorld()

    # Mock LLM that recommends action 0 (usually "look")
    mock_llm = LLMAdvisorTool(generate_fn=lambda _: "0")
    tools = list(DEFAULT_TOOLS) + [mock_llm]

    agent = IFAgent(world=world, tools=tools)
    result = agent.play_game(max_steps=20)

    assert result.steps_taken > 0
    assert result.reliability_table is not None


def test_agent_verbose_mode(capsys):
    """Verbose mode should print step-by-step trace."""
    world = MockWorld()
    agent = IFAgent(world=world, tools=list(DEFAULT_TOOLS), verbose=True)
    agent.play_game(max_steps=5)

    captured = capsys.readouterr()
    assert "[Step 1]" in captured.out


def test_reliability_table_updates():
    """After playing, reliability table should differ from uniform Beta(1,1)."""
    world = MockWorld()

    # Use an LLM that sometimes gives good advice
    call_count = 0
    def mock_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # Return "take key" index early, then "go north"
        return str(call_count % 3)

    tools = list(DEFAULT_TOOLS) + [LLMAdvisorTool(generate_fn=mock_llm)]
    agent = IFAgent(world=world, tools=tools)
    result = agent.play_game(max_steps=30)

    # At least some reliability entries should have moved from Beta(1,1)
    table = result.reliability_table
    assert table is not None
    # Check that not all entries are still (1.0, 1.0)
    deviations = np.abs(table[:, :, 0] - 1.0) + np.abs(table[:, :, 1] - 1.0)
    assert np.any(deviations > 0.01), "Reliability table should have been updated"


def test_agent_records_steps():
    """Each step should be recorded with valid fields."""
    world = MockWorld()
    agent = IFAgent(world=world)
    result = agent.play_game(max_steps=10)

    for step in result.steps:
        assert step.step >= 1
        assert isinstance(step.chosen_action, str)
        assert isinstance(step.confidence, float)
        assert 0.0 <= step.confidence <= 1.0


def test_play_step_with_no_valid_actions():
    """Agent should handle empty valid_actions gracefully."""
    world = MockWorld()
    obs = world.reset()

    # Win the game first
    world.step("take key")
    world.step("go north")
    world.step("go north")
    world.step("open chest")

    # Now the game is done — but if we call valid_actions, it might still return some
    # Just test that IFAgent handles empty lists
    agent = IFAgent(world=world)
    # Create a patched world that returns no actions


    def empty_actions() -> list[str]:
        return []

    world.valid_actions = empty_actions  # type: ignore[assignment]
    action, record = agent.play_step(obs)
    assert action == "look"  # safe fallback
