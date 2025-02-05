import pytest
import numpy as np
from myrtle.agents.random_multi_action import RandomMultiAction

np.random.seed(42)

_n_sensors = 6
_n_actions = 5
_n_rewards = 4


@pytest.fixture
def initialize_agent():
    agent = RandomMultiAction(
        n_sensors=_n_sensors,
        n_actions=_n_actions,
        n_rewards=_n_rewards,
    )

    yield agent

    agent.close()


def test_action_selection(initialize_agent):
    agent = initialize_agent

    passed = False
    n_tries = 10
    for _ in range(n_tries):
        agent.choose_action()
        if sum(agent.actions) > 1:
            passed = True
            break
    assert passed
