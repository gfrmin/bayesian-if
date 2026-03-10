"""Tests for reward attribution."""

from bayesian_if.reward import attribute_reward
from bayesian_if.world import Observation


def test_positive_reward_is_correct():
    assert attribute_reward(5.0) is True
    assert attribute_reward(0.1) is True


def test_negative_reward_is_wrong():
    assert attribute_reward(-1.0) is False
    assert attribute_reward(-0.5) is False


def test_zero_reward_is_ambiguous():
    assert attribute_reward(0.0) is None


def test_zero_reward_same_obs_is_wrong():
    obs = Observation(text="A dark room.", score=0, location="room", inventory=("key",))
    assert attribute_reward(0.0, obs, obs) is False


def test_zero_reward_different_location_is_ambiguous():
    prev = Observation(text="A dark room.", score=0, location="room1", inventory=())
    new = Observation(text="A bright room.", score=0, location="room2", inventory=())
    assert attribute_reward(0.0, prev, new) is None


def test_zero_reward_different_inventory_is_ambiguous():
    prev = Observation(text="A dark room.", score=0, location="room", inventory=())
    new = Observation(text="A dark room.", score=0, location="room", inventory=("key",))
    assert attribute_reward(0.0, prev, new) is None


def test_no_observations_backward_compat():
    assert attribute_reward(0.0) is None
    assert attribute_reward(0.0, None, None) is None
