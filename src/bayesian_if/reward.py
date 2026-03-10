"""Reward attribution: bridge IF score deltas to credence's was_correct signal."""

from __future__ import annotations

from bayesian_if.world import Observation


def _obs_unchanged(prev: Observation, new: Observation) -> bool:
    """True if the observation represents no meaningful state change.

    Compares location and inventory (structured state). Falls back to text
    comparison only when location is unavailable for both observations.
    """
    if prev.location is not None or new.location is not None:
        return prev.location == new.location and prev.inventory == new.inventory
    return prev.text == new.text and prev.inventory == new.inventory


def attribute_reward(
    score_delta: float,
    prev_obs: Observation | None = None,
    new_obs: Observation | None = None,
) -> bool | None:
    """Convert a score delta into a correctness signal for reliability updates.

    - score_delta > 0  → True  (action led to progress)
    - score_delta < 0  → False (action was harmful)
    - score_delta == 0 + state unchanged → False (wasted turn)
    - score_delta == 0 + state changed   → None  (ambiguous)
    - score_delta == 0 + no observations → None  (backward compatible)
    """
    if score_delta > 0:
        return True
    elif score_delta < 0:
        return False
    elif prev_obs is not None and new_obs is not None:
        return False if _obs_unchanged(prev_obs, new_obs) else None
    else:
        return None
