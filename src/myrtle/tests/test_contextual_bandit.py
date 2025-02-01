import pytest
import numpy as np
from myrtle.worlds import contextual_bandit


@pytest.fixture
def initialize_world():
    world = contextual_bandit.ContextualBandit()

    yield world


def test_initialization(initialize_world):
    world = initialize_world
    assert world.n_sensors == 4
    assert world.n_actions == 4
    assert world.n_rewards == 4

    assert world.bandit_payouts[2] == 280
    assert world.bandit_hit_rates[1] == 0.25


def test_sense(initialize_world):
    world = initialize_world

    world.i_loop_step = 0
    world.sense()
    assert world.bandit_order[1] == world.sensors[1]
    assert world.bandit_order[2] == world.sensors[2]

    n_tries = 1000
    sum_order = 0.0
    for _ in range(n_tries):
        world.sense()
        sum_order += world.bandit_order[2]
    mean_order = sum_order / n_tries
    # Should be ~1.5 +/- some variance
    assert mean_order > 1.3
    assert mean_order < 1.7


def test_step_world(initialize_world):
    world = initialize_world

    world.actions = np.zeros(world.n_actions)
    world.actions[1] = 1

    n_tries = 1000
    sum_reward = 0.0
    for _ in range(n_tries):
        world.sense()
        world.step_world()
        sum_reward += np.sum(world.rewards)
    mean_reward = sum_reward / n_tries
    # Should be ~64 +/- some variance
    print(mean_reward)
    assert mean_reward > 58
    assert mean_reward < 70
