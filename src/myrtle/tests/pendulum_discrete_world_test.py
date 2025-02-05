import pytest
import numpy as np
from myrtle.worlds import pendulum_discrete


@pytest.fixture
def initialize_world():
    world = pendulum_discrete.PendulumDiscrete()

    yield world


def test_reset_sensors(initialize_world):
    world = initialize_world
    world.reset_sensors()

    assert world.n_positions == 36
    assert world.n_velocities == 62
    assert np.sum(world.sensors) == 0.0


def test_step_sensors(initialize_world):
    world = initialize_world
    world.reset_sensors()
    world.step_sensors()

    assert np.sum(world.sensors) == 2
