"""Reward attribution: bridge IF score deltas to credence's was_correct signal."""

from __future__ import annotations


def attribute_reward(score_delta: float) -> bool | None:
    """Convert a score delta into a correctness signal for reliability updates.

    - score_delta > 0  → True  (action led to progress)
    - score_delta < 0  → False (action was harmful)
    - score_delta == 0 → None  (ambiguous, no reliability update)
    """
    if score_delta > 0:
        return True
    elif score_delta < 0:
        return False
    else:
        return None
