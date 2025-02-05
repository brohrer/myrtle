import pytest
import time
import numpy as np
from myrtle.worlds import stationary_bandit

np.random.seed(42)
_log_name = f"world_{int(time.time())}"


@pytest.fixture
def initialize_world():
    world = stationary_bandit.StationaryBandit()

    yield world


def test_initialization(initialize_world):
    world = initialize_world
    assert world.n_sensors == 0
    assert world.n_actions == 5
    assert world.n_rewards == 5


def test_sense(initialize_world):
    world = initialize_world

    world.actions = np.zeros(world.n_actions)
    world.actions[2] = 1
    hit_reward_2 = world.actions[2] * world.bandit_payouts[2]
    assert hit_reward_2 == 280

    n_tries = 1000
    sum_reward = 0.0
    for _ in range(n_tries):
        world.sense()
        sum_reward += np.sum(world.rewards)
    mean_reward = sum_reward / n_tries
    # Should be 112 +/- some variance
    assert mean_reward > 106.0
    assert mean_reward < 118.0
