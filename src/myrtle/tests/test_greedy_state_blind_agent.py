import pytest
import numpy as np
from myrtle.agents.greedy_state_blind import GreedyStateBlind

np.random.seed(42)

_n_sensors = 7
_n_actions = 5
_n_rewards = 3


@pytest.fixture
def initialize_agent():
    agent = GreedyStateBlind(
        n_sensors=_n_sensors,
        n_actions=_n_actions,
        n_rewards=_n_rewards,
    )

    yield agent

    agent.close()


def test_action_selection(initialize_agent):
    agent = initialize_agent
    agent.actions = np.zeros(agent.n_actions)
    agent.actions[2] = 1.0
    agent.rewards = 0.5 * np.ones(agent.n_rewards)
    agent.total_return = np.ones(agent.n_actions)
    agent.action_count = np.ones(agent.n_actions)
    agent.choose_action()
    assert np.sum(agent.actions) == 1


def test_reset(initialize_agent):
    agent = initialize_agent
    agent.total_return = 13 * np.ones(agent.n_actions)
    agent.action_count = 13 * np.ones(agent.n_actions)
    agent.reset()
    assert agent.total_return[2] == 1.0
    assert agent.action_count[3] == 1.0
