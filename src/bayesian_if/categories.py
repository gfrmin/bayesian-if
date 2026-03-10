"""IF situation categories and category inference function."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from numpy.typing import NDArray

CATEGORIES: tuple[str, ...] = ("exploration", "puzzle", "inventory", "dialogue", "combat")
NUM_CATEGORIES: int = len(CATEGORIES)

# Keyword patterns per category (index-aligned with CATEGORIES).
_CATEGORY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "exploration": [
        re.compile(r"\b(dark|passage|room|door|north|south|east|west|up|down|corridor|hall)\b", re.I),
    ],
    "puzzle": [
        re.compile(r"\b(locked|key|lever|button|switch|mechanism|puzzle|open|close|insert)\b", re.I),
    ],
    "inventory": [
        re.compile(r"\b(take|drop|carry|holding|pick up|put down|wearing|remove)\b", re.I),
    ],
    "dialogue": [
        re.compile(r"\b(says?|asks?|tells?|speak|talk|reply|replies|greet|hello)\b", re.I),
    ],
    "combat": [
        re.compile(r"\b(attack|kill|fight|sword|troll|monster|hit|slash|wound|dead)\b", re.I),
    ],
}


def make_if_category_infer_fn(
    categories: tuple[str, ...] = CATEGORIES,
) -> Callable[[str], NDArray[np.float64]]:
    """Return a function that classifies game-state text into a category distribution.

    Uses keyword matching with a Dirichlet-like weighting: each keyword match
    adds weight to the corresponding category, then normalise.
    """

    def infer(text: str) -> NDArray[np.float64]:
        weights = np.ones(len(categories), dtype=np.float64)  # uniform base
        for i, cat in enumerate(categories):
            patterns = _CATEGORY_PATTERNS.get(cat, [])
            for pat in patterns:
                matches = pat.findall(text)
                weights[i] += len(matches) * 2.0  # each match boosts that category
        total = weights.sum()
        return weights / total

    return infer
