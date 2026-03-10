"""IFAgent — composes BayesianAgent with IF domain logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from credence import BayesianAgent, ScoringRule

from bayesian_if.categories import CATEGORIES, make_if_category_infer_fn
from bayesian_if.reward import attribute_reward
from bayesian_if.tools import DEFAULT_TOOLS, IFTool
from bayesian_if.world import Observation, World

# Mild asymmetry: wrong IF actions waste a turn but aren't catastrophic.
IF_SCORING = ScoringRule(reward_correct=1.0, penalty_wrong=-0.2, reward_abstain=0.0)


@dataclass
class StepRecord:
    """Record of a single game step."""

    step: int
    observation_text: str
    valid_actions: list[str]
    chosen_action: str
    tools_queried: tuple[int, ...]
    confidence: float
    reward: float
    cumulative_score: int


@dataclass
class GameResult:
    """Result of playing a full game."""

    final_score: int
    steps_taken: int
    steps: list[StepRecord] = field(default_factory=list)
    reliability_table: NDArray[np.float64] | None = None


class IFAgent:
    """Bayesian decision-theoretic IF agent.

    Uses BayesianAgent as the information-gathering controller: each game step,
    VOI decides which info sources to consult before committing to an action.
    The reliability table persists across steps, learning which sources work
    in which situations.
    """

    def __init__(
        self,
        world: World,
        tools: list[IFTool] | None = None,
        categories: tuple[str, ...] = CATEGORIES,
        category_infer_fn: Callable[[str], NDArray] | None = None,
        scoring: ScoringRule = IF_SCORING,
        forgetting: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.world = world
        self.if_tools = tools if tools is not None else list(DEFAULT_TOOLS)
        self.categories = categories
        self.scoring = scoring
        self.verbose = verbose
        self._history: list[tuple[str, str]] = []

        if category_infer_fn is None:
            category_infer_fn = make_if_category_infer_fn(categories)

        tool_configs = [t.to_tool_config(categories) for t in self.if_tools]

        self.bayesian = BayesianAgent(
            tool_configs=tool_configs,
            categories=categories,
            category_infer_fn=category_infer_fn,
            forgetting=forgetting,
        )

    def play_step(self, observation: Observation) -> tuple[str, StepRecord]:
        """One game step: gather info via VOI, choose action, return action string."""
        valid_actions = self.world.valid_actions()

        if not valid_actions:
            return "look", StepRecord(
                step=0, observation_text=observation.text,
                valid_actions=[], chosen_action="look",
                tools_queried=(), confidence=0.0,
                reward=0.0, cumulative_score=observation.score,
            )

        # Build the tool query function that BayesianAgent will call
        recent_history = self._history[-5:] if self._history else None

        def tool_query_fn(tool_idx: int) -> int | None:
            return self.if_tools[tool_idx].query(
                self.world, observation, valid_actions, history=recent_history
            )

        result = self.bayesian.solve_question(
            question_text=observation.text,
            candidates=tuple(valid_actions),
            category_hint=None,
            tool_query_fn=tool_query_fn,
        )

        if result.answer is not None and result.answer < len(valid_actions):
            chosen_action = valid_actions[result.answer]
        else:
            # Abstain → take a safe action
            chosen_action = _safe_action(valid_actions)

        record = StepRecord(
            step=0,  # filled in by play_game
            observation_text=observation.text,
            valid_actions=valid_actions,
            chosen_action=chosen_action,
            tools_queried=result.tools_queried,
            confidence=result.confidence,
            reward=0.0,  # filled in after step
            cumulative_score=observation.score,
        )

        return chosen_action, record

    def play_game(self, max_steps: int = 100) -> GameResult:
        """Play a full game, returning trace and final score."""
        obs = self.world.reset()
        self._history = []
        steps: list[StepRecord] = []

        for step_num in range(1, max_steps + 1):
            action, record = self.play_step(obs)
            prev_obs = obs
            obs, reward, done = self.world.step(action)

            # Track history for LLM context
            self._history.append((action, obs.text[:100]))

            # Attribute reward for reliability learning
            was_correct = attribute_reward(reward, prev_obs, obs)
            self.bayesian.on_question_end(was_correct)

            record.step = step_num
            record.reward = reward
            record.cumulative_score = obs.score
            steps.append(record)

            if self.verbose:
                tools_str = ", ".join(self.if_tools[t].name for t in record.tools_queried)
                print(
                    f"[Step {step_num}] "
                    f"Action: {action!r}  "
                    f"Tools: [{tools_str}]  "
                    f"Confidence: {record.confidence:.2f}  "
                    f"Reward: {reward:+.0f}  "
                    f"Score: {obs.score}"
                )

            if done:
                break

        return GameResult(
            final_score=obs.score,
            steps_taken=len(steps),
            steps=steps,
            reliability_table=self.bayesian.reliability_table.copy(),
        )


def _safe_action(valid_actions: list[str]) -> str:
    """Pick a safe fallback action when the agent abstains."""
    safe_verbs = ("look", "wait", "inventory", "examine")
    for action in valid_actions:
        if any(action.lower().startswith(v) for v in safe_verbs):
            return action
    return valid_actions[0]
