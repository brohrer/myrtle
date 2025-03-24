import multiprocessing as mp
import pytest
import numpy as np
from myrtle.tests.fixtures import setup_mq_server  # noqa: F401
from myrtle.worlds import pendulum_discrete


@pytest.fixture
def initialize_world():
    world = pendulum_discrete.PendulumDiscrete(
        q_action=mp.Queue(),
        q_reward=mp.Queue(),
        q_sensor=mp.Queue(),
    )

    yield world


def test_reset_sensors(initialize_world):
    world = initialize_world
    world.reset_sensors()

    assert world.n_positions == 36
    assert world.n_velocities == 62
    assert np.sum(world.sensors) == 0.0


def test_step_sensors(
    initialize_world,
    setup_mq_server,  # noqa: F811
):
    world = initialize_world
    world.initialize_mq()
    world.reset_sensors()
    world.step_sensors()

    assert np.sum(world.sensors) == 2
