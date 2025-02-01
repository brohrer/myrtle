import pytest
import numpy as np
from myrtle.worlds import intermittent_reward_bandit


@pytest.fixture
def initialize_world():
    world = intermittent_reward_bandit.IntermittentRewardBandit()

    yield world


def test_initialization(initialize_world):
    world = initialize_world
    assert world.n_sensors == 0
    assert world.n_actions == 5
    assert world.n_rewards == 5


def test_sense(initialize_world):
    world = initialize_world

    world.i_loop_step = 0
    world.actions = np.zeros(world.n_actions)
    world.actions[2] = 1
    hit_reward_2 = world.actions[2] * world.bandit_payouts[2]
    assert hit_reward_2 == 280

    n_tries = 1000
    sum_reward = 0.0
    for _ in range(n_tries):
        world.sense()
        reward_total = 0
        for r in world.rewards:
            if r is not None:
                reward_total += r
        sum_reward += reward_total
    mean_reward = sum_reward / n_tries
    # Should be ~100 +/- some variance
    assert mean_reward > 90.0
    assert mean_reward < 110.0


def test_intermittency(initialize_world):
    world = initialize_world

    world.actions = np.zeros(world.n_actions)
    world.actions[1] = 1

    n_tries = 100
    found_none = False
    for _ in range(n_tries):
        world.sense()
        if world.rewards[1] is None:
            found_none = True
    assert found_none
