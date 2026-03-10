"""Information-gathering tools for IF — each returns a recommended action index."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from credence import ToolConfig

from bayesian_if.categories import CATEGORIES
from bayesian_if.world import Observation, World

# Known IF verbs for action parsing.
IF_VERBS = frozenset({
    "go", "take", "open", "examine", "push", "pull", "turn", "read",
    "drop", "put", "close", "unlock", "insert", "look", "inventory",
    "wait", "eat", "drink", "wear", "remove", "give", "ask", "tell",
    "attack", "tie", "cut", "climb", "enter", "exit", "search",
})


class IFTool(ABC):
    """Base class for IF information-gathering tools."""

    name: str
    cost: float

    @abstractmethod
    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> int | None:
        """Query this tool and return a recommended action index, or None."""
        ...

    def to_tool_config(self, categories: tuple[str, ...] = CATEGORIES) -> ToolConfig:
        """Convert to a credence ToolConfig with per-category coverage."""
        return ToolConfig(
            cost=self.cost,
            coverage_by_category=self._coverage(categories),
        )

    @abstractmethod
    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        """Return P(tool returns an answer | category) for each category."""
        ...


# ---------------------------------------------------------------------------
# Phase 3: Principled action matching
# ---------------------------------------------------------------------------

def _parse_action(action: str) -> tuple[str | None, list[str]]:
    """Split an IF action into (verb, object_words).

    Known IF verbs are recognised; everything after the verb is objects.
    If the first word is not a known verb, verb is None and all words are objects.
    """
    words = action.lower().split()
    if not words:
        return None, []
    verb = words[0] if words[0] in IF_VERBS else None
    objects = words[1:] if verb else words
    return verb, objects


def _score_actions(
    valid_actions: list[str], verb: str | None, nouns: list[str]
) -> int | None:
    """Score each valid action against a recommended verb + nouns.

    - Verb match: +3.0 (high weight — "take key" beats "examine key" when tool said "take")
    - Object match: +1.0 per noun, using word-boundary matching (\\bkey\\b not substring)
    - Returns argmax index if any action scores > 0, else None.
    """
    best_idx: int | None = None
    best_score = 0.0
    for i, action in enumerate(valid_actions):
        a_verb, a_objects = _parse_action(action)
        score = 0.0
        if verb and a_verb == verb:
            score += 3.0
        obj_text = " ".join(a_objects)
        for noun in nouns:
            if re.search(r"\b" + re.escape(noun) + r"\b", obj_text, re.I):
                score += 1.0
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _best_action_matching(
    valid_actions: list[str], keywords: list[str]
) -> int | None:
    """Return the index of the valid action best matching the keywords, or None."""
    best_idx: int | None = None
    best_score = 0
    for i, action in enumerate(valid_actions):
        action_lower = action.lower()
        score = sum(1 for kw in keywords if kw.lower() in action_lower)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _extract_keywords(text: str) -> list[str]:
    """Extract notable words from descriptive text."""
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "to", "of", "and", "you", "it", "that", "this",
    }
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in stop and len(w) > 2]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class LookTool(IFTool):
    """Peek at 'look' output in a save/restore bracket."""

    name = "look"
    cost = 0.0

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> int | None:
        snapshot = world.save()
        try:
            obs, _, _ = world.step("look")
            keywords = _extract_keywords(obs.text)
            # Phase 1: incorporate location for directional disambiguation
            if observation.location:
                keywords.append(observation.location.lower())
            return _score_actions(valid_actions, verb=None, nouns=keywords)
        finally:
            world.restore(snapshot)

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        # Look is most useful for exploration, somewhat for puzzle
        coverage = {
            "exploration": 0.9,
            "puzzle": 0.5,
            "inventory": 0.3,
            "dialogue": 0.4,
            "combat": 0.4,
        }
        return np.array([coverage.get(c, 0.5) for c in categories])


class ExamineTool(IFTool):
    """Examine the most novel visible object in a save/restore bracket."""

    name = "examine"
    cost = 0.0

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> int | None:
        # Find an object to examine from valid actions or observation text
        target = self._pick_target(observation, valid_actions)
        if target is None:
            return None

        snapshot = world.save()
        try:
            obs, _, _ = world.step(f"examine {target}")
            keywords = _extract_keywords(obs.text)
            keywords.append(target)
            return _score_actions(valid_actions, verb=None, nouns=keywords)
        finally:
            world.restore(snapshot)

    @staticmethod
    def _pick_target(observation: Observation, valid_actions: list[str]) -> str | None:
        """Pick an object to examine — prefer items mentioned in actions."""
        # Look for nouns in valid actions that suggest examinable objects
        for action in valid_actions:
            match = re.match(
                r"(?:take|examine|open|push|pull|turn|read)\s+(.+)", action, re.I
            )
            if match:
                return match.group(1).strip()
        # Phase 1: consider inventory items as examination targets
        if observation.inventory:
            return observation.inventory[0]
        # Fallback: look for nouns in observation text
        nouns = re.findall(r"\b([A-Z][a-z]+)\b", observation.text)
        return nouns[0].lower() if nouns else None

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        coverage = {
            "exploration": 0.5,
            "puzzle": 0.9,
            "inventory": 0.6,
            "dialogue": 0.3,
            "combat": 0.3,
        }
        return np.array([coverage.get(c, 0.5) for c in categories])


class InventoryTool(IFTool):
    """Check inventory in a save/restore bracket."""

    name = "inventory"
    cost = 0.0

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> int | None:
        # Phase 1: use structured inventory when available
        if observation.inventory:
            items = [item.lower() for item in observation.inventory]
            return _score_actions(valid_actions, verb=None, nouns=items)
        # Fallback: save/restore bracket
        snapshot = world.save()
        try:
            obs, _, _ = world.step("inventory")
            keywords = _extract_keywords(obs.text)
            return _score_actions(valid_actions, verb=None, nouns=keywords)
        finally:
            world.restore(snapshot)

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        coverage = {
            "exploration": 0.2,
            "puzzle": 0.5,
            "inventory": 0.9,
            "dialogue": 0.1,
            "combat": 0.4,
        }
        return np.array([coverage.get(c, 0.3) for c in categories])


class LLMAdvisorTool(IFTool):
    """Ask an LLM which action to take."""

    name = "llm_advisor"
    cost = 1.0

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        model: str = "llama3.2",
    ) -> None:
        if generate_fn is not None:
            self._generate = generate_fn
        else:
            from bayesian_if.ollama import ollama_generate

            self._generate = lambda prompt: ollama_generate(prompt, model=model)

    def query(
        self,
        world: World,
        observation: Observation,
        valid_actions: list[str],
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> int | None:
        if not valid_actions:
            return None

        actions_str = "\n".join(f"  {i}: {a}" for i, a in enumerate(valid_actions))

        # Phase 2: build rich context from structured observation + history
        context_parts: list[str] = []
        if observation.location:
            context_parts.append(f"Current location: {observation.location}")
        if observation.inventory:
            context_parts.append(f"Inventory: {', '.join(observation.inventory)}")
        if observation.objective:
            context_parts.append(f"Objective: {observation.objective}")
        if history:
            history_lines = [f"  > {act} -> {res}" for act, res in history[-5:]]
            context_parts.append("Recent history:\n" + "\n".join(history_lines))

        context = "\n".join(context_parts)

        prompt = f"You are playing a text adventure game.\n\nCurrent situation:\n{observation.text}\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += (
            f"Available actions:\n{actions_str}\n\n"
            f"Which action number is best? Reply with ONLY the number."
        )

        try:
            response = self._generate(prompt)
            # Extract first integer from response
            match = re.search(r"\d+", response)
            if match:
                idx = int(match.group())
                if 0 <= idx < len(valid_actions):
                    return idx
        except Exception:
            pass
        return None

    def _coverage(self, categories: tuple[str, ...]) -> np.ndarray:
        # LLM has broad but imperfect coverage — the agent learns the truth
        return np.full(len(categories), 0.7)


DEFAULT_TOOLS: list[IFTool] = [LookTool(), ExamineTool(), InventoryTool()]
"""Default tools (no LLM). Add LLMAdvisorTool separately when Ollama is available."""
