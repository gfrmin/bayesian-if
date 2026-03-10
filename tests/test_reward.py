"""Tests for reward attribution."""

from bayesian_if.reward import attribute_reward


def test_positive_reward_is_correct():
    assert attribute_reward(5.0) is True
    assert attribute_reward(0.1) is True


def test_negative_reward_is_wrong():
    assert attribute_reward(-1.0) is False
    assert attribute_reward(-0.5) is False


def test_zero_reward_is_ambiguous():
    assert attribute_reward(0.0) is None
