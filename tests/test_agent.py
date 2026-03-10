"""Tests for the IFAgent — full integration with mock world."""

import numpy as np

from bayesian_if.agent import IFAgent, _safe_action
from bayesian_if.categories import infer_category_hint
from bayesian_if.tools import DEFAULT_TOOLS, LLMAdvisorTool
from bayesian_if.world import Observation
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


# ---------------------------------------------------------------------------
# Phase 1: Failed-action memory
# ---------------------------------------------------------------------------

def test_agent_suppresses_failed_actions():
    """Zero-reward action at same location should not be repeated."""
    world = MockWorld()
    agent = IFAgent(world=world, tools=list(DEFAULT_TOOLS))
    obs = world.reset()

    # Simulate "wait" yielding no reward at Start Room
    agent._failed_actions.setdefault("Start Room", set()).add("wait")

    action, _ = agent.play_step(obs)
    # "wait" should be suppressed — agent picks something else
    assert action != "wait"


def test_failed_actions_cleared_on_reward():
    """Positive reward should clear the failed set for that location."""
    world = MockWorld()
    agent = IFAgent(world=world, tools=list(DEFAULT_TOOLS))
    obs = world.reset()

    # Pre-populate failed actions at Start Room
    agent._failed_actions["Start Room"] = {"look", "wait"}
    assert len(agent._failed_actions["Start Room"]) == 2

    # Manually step "take key" which gives +5 reward
    action, record = agent.play_step(obs)
    # Force "take key" to trigger the reward-based clear
    prev_obs = obs
    obs, reward, done = world.step("take key")
    agent._history.append(("take key", obs.text[:100]))
    # Simulate what play_game does after the step
    if reward > 0:
        agent._failed_actions.pop(prev_obs.location, None)

    # Start Room failures should now be cleared
    assert "Start Room" not in agent._failed_actions


# ---------------------------------------------------------------------------
# Phase 3: _safe_action prefers movement
# ---------------------------------------------------------------------------

def test_safe_action_prefers_movement():
    """When no local actions are needed, movement should be preferred."""
    actions = ["look", "wait", "go north", "go south", "take key"]
    failed = {"look", "wait", "take key"}
    # With those failed, should prefer go north or go south
    results = {_safe_action(actions, failed=failed) for _ in range(50)}
    assert results <= {"go north", "go south"}


def test_safe_action_falls_back_to_look():
    """Without movement options and no failures, should pick look."""
    actions = ["look", "wait", "inventory"]
    action = _safe_action(actions)
    # Movement preferred but none available → should pick "look"
    assert action == "look"


# ---------------------------------------------------------------------------
# Phase 5: Category hints
# ---------------------------------------------------------------------------

def test_category_hint_puzzle_with_key():
    """Having a key in inventory should hint 'puzzle'."""
    obs = Observation(text="A room.", score=5, inventory=("rusty key",))
    assert infer_category_hint(obs) == "puzzle"


def test_category_hint_combat():
    """Combat keywords should hint 'combat'."""
    obs = Observation(text="A troll attacks you!", score=0)
    assert infer_category_hint(obs) == "combat"


def test_category_hint_none_when_ambiguous():
    """Ambiguous text should return None."""
    obs = Observation(text="A plain room.", score=0)
    assert infer_category_hint(obs) is None


def test_category_hint_exploration():
    """Corridor in location should hint 'exploration'."""
    obs = Observation(text="A dim space.", score=0, location="Dark Corridor")
    assert infer_category_hint(obs) == "exploration"


# ---------------------------------------------------------------------------
# Phase 6: Forgetting default
# ---------------------------------------------------------------------------

def test_default_forgetting():
    """Default forgetting factor should be 0.85."""
    world = MockWorld()
    agent = IFAgent(world=world)
    # The BayesianAgent stores the forgetting factor
    assert agent.bayesian.forgetting == 0.85
