import pytest
import numpy as np
from myrtle.agents.q_learning_curiosity import QLearningCuriosity

_pause = 0.01  # seconds

np.random.seed(42)

_n_sensors = 8
_n_actions = 7
_n_rewards = 5


@pytest.fixture
def initialize_agent():
    agent = QLearningCuriosity(
        n_sensors=_n_sensors,
        n_actions=_n_actions,
        n_rewards=_n_rewards,
    )

    yield agent

    agent.close()


def test_creation(initialize_agent):
    agent = initialize_agent
    assert agent.n_sensors == _n_sensors
    assert agent.n_actions == _n_actions
    assert agent.n_rewards == _n_rewards


def test_learning_rate_updating(initialize_agent):
    agent = initialize_agent
    agent.epsilon = 0.0
    agent.discount_factor = 0.0
    agent.learning_rate = 0.5
    agent.reset()

    agent.actions = np.array([0, 1, 0])
    agent.rewards = np.array([0, 64])
    agent.i_step = 0

    agent.choose_action()

    assert agent.q_values[agent.previous_sensors.tobytes()][0] == 0
    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 64
    assert agent.q_values[agent.previous_sensors.tobytes()][2] == 0

    agent.rewards = np.array([0, 128])
    agent.choose_action()

    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 96

    agent.choose_action()

    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 112


def test_discount_factor_updating(initialize_agent):
    agent = initialize_agent
    agent.epsilon = 0.0
    agent.discount_factor = 0.5
    agent.learning_rate = 0.5
    agent.reset()

    # agent.previous_sensors = np.array([1, 2, 3, 4])
    # agent.sensors = np.array([1, 2, 3, 4])
    # agent.q_values[agent.previous_sensors.tobytes()] = np.zeros(agent.n_actions)
    agent.q_values[agent.sensors.tobytes()] = np.ones(agent.n_actions) * 100
    previous_action = 2
    agent.counts[agent.previous_sensors.tobytes()] = np.zeros(agent.n_actions)
    agent.counts[agent.previous_sensors.tobytes()][previous_action] = 2
    agent.actions = np.array([0, 1, 0])
    agent.rewards = np.array([0, 12])
    agent.i_step = 0

    agent.choose_action()

    assert agent.q_values[agent.previous_sensors.tobytes()][0] == 100
    assert agent.q_values[agent.previous_sensors.tobytes()][1] == 62
    assert agent.q_values[agent.previous_sensors.tobytes()][2] == 100
