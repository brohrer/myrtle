import json
import pytest
from threading import Thread
import time
import numpy as np
from myrtle.agents import base_agent
from myrtle.tests.world_mocks import (
    multiepisode_world,
    world_step_random,
)

# Exclude pytest fixtures from some checks because they behave in peculiar ways.
from myrtle.tests.fixtures import setup_mq_server, setup_mq_client  # noqa: F401

np.random.seed(42)
# times in seconds
_pause = 0.01
_long_pause = 0.1
_max_retries = 9

_n_sensors = 5
_n_actions = 4
_n_rewards = 3


@pytest.fixture
def initialize_agent():
    agent = base_agent.BaseAgent(
        n_sensors=_n_sensors,
        n_actions=_n_actions,
        n_rewards=_n_rewards,
    )

    yield agent

    agent.close()


def test_initialization(
    setup_mq_server,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    assert agent.n_sensors == _n_sensors
    assert agent.n_actions == _n_actions
    assert agent.n_rewards == _n_rewards


def test_action_generation(
    setup_mq_server,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    agent.choose_action()

    # There should be just one nonzero action, and it should have a value of 1.
    assert agent.actions.size == _n_actions
    assert np.where(agent.actions > 0)[0].size == 1
    assert np.sum(agent.actions) == 1.0


def test_reset(
    setup_mq_server,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    agent.choose_action()
    agent.reset()
    time.sleep(_pause)
    assert np.sum(agent.actions) == 0


def test_world_step_read(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    agent.initialize_mq()
    agent.reset()
    mq = setup_mq_client

    world_value = world_step_random(mq, _n_sensors, _n_rewards)
    time.sleep(_pause)
    agent.initialize_mq()
    agent.read_world_step()
    assert agent.sensors[_n_sensors - 1] == world_value
    assert agent.rewards[_n_rewards - 1] == world_value


def test_action_write(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    mq = setup_mq_client
    agent.i_step = 0
    agent.i_episode = 0
    agent.initialize_mq()
    agent.choose_action()
    agent.write_agent_step()

    agent_info = json.loads(mq.get_wait("agent_step"))
    assert agent_info["episode"] == 0
    assert agent_info["step"] == 0


def test_control_check(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    mq = setup_mq_client
    agent.initialize_mq()
    mq.put("control", "truncated")
    time.sleep(_pause)
    episode_complete, run_complete = agent.control_check()

    assert episode_complete
    assert not run_complete

    mq.put("control", "terminated")
    time.sleep(_pause)
    episode_complete, run_complete = agent.control_check()

    assert not episode_complete
    assert run_complete


def test_episode_advancement(
    setup_mq_server,  # noqa: F811
    setup_mq_client,  # noqa: F811
    initialize_agent,
):
    agent = initialize_agent
    mq = setup_mq_client
    world = Thread(target=multiepisode_world, args=(mq, _n_sensors, _n_rewards))
    world.start()
    agent.run()

    # For this join() to complete, the "terminated" signal will also
    # need to be read and interpreted correctly.
    # It's an indirect test.
    world.join()

    # Get the most recent agent_step message
    response = mq.get("agent_step")
    while response != "":
        msg_str = response
        response = mq.get("agent_step")

    agent_info = json.loads(msg_str)
    i_episode = agent_info["episode"]

    assert i_episode == 1
