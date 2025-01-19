import json
import multiprocessing as mp
import pytest
import signal
import time
import numpy as np
import dsmq.client
# from sqlogging import logging
from myrtle.agents import base_agent
# from myrtle.config import LOG_DIRECTORY
from myrtle.tests.fixtures import setup_mq_server, setup_mq_client

np.random.seed(42)
# times in seconds
_pause = 0.01
_long_pause = 0.3
_timeout = 1.0
_max_retries = 9

_n_sensors = 5
_n_actions = 4
_n_rewards = 3


def initialize_agent():
    return base_agent.BaseAgent(
        n_sensors=_n_sensors,
        n_actions=_n_actions,
        n_rewards=_n_rewards,
    )


def world_step_fake(mq, value=0):
    i_loop_step = int(value)
    i_episode = int(value)
    sensors = np.ones(_n_sensors) * value
    rewards = np.ones(_n_rewards) * value
    world_step_msg = json.dumps(
        {
            "loop_step": i_loop_step,
            "episode": i_episode,
            "sensors": sensors.tolist(),
            "rewards": rewards.tolist(),
        }
    )
    mq.put("world_step", world_step_msg)
    return value


def world_step_random(mq):
    return world_step_fake(mq, value=np.random.choice(17))


def world_step_zero(mq):
    i_loop_step = 0
    i_episode = 0
    sensors = np.zeros(_n_sensors)
    rewards = np.zeros(_n_rewards)
    world_step_msg = json.dumps(
        {
            "loop_step": i_loop_step,
            "episode": i_episode,
            "sensors": sensors.tolist(),
            "rewards": rewards.tolist(),
        }
    )
    mq.put("world_step", world_step_msg)


def test_initialization(setup_mq_server):
    agent = initialize_agent()
    assert agent.n_sensors == _n_sensors
    assert agent.n_actions == _n_actions
    assert agent.n_rewards == _n_rewards


def test_action_generation(setup_mq_server):
    agent = initialize_agent()
    agent.choose_action()

    # There should be just one nonzero action, and it should have a value of 1.
    assert agent.actions.size == _n_actions
    assert np.where(agent.actions > 0)[0].size == 1
    assert np.sum(agent.actions) == 1.0


def test_reset(setup_mq_server):
    agent = initialize_agent()
    agent.choose_action()
    agent.reset()
    time.sleep(_pause)
    assert np.sum(agent.actions) == 0


def test_world_step_read(setup_mq_server, setup_mq_client):
    agent = initialize_agent()
    mq = setup_mq_client

    world_value = world_step_random(mq)
    time.sleep(_pause)
    agent.read_world_step()
    assert agent.sensors[_n_sensors - 1] == world_value
    assert agent.rewards[_n_rewards - 1] == world_value


def test_action_write(setup_mq_server, setup_mq_client):
    agent = initialize_agent()
    agent.i_step = 0
    agent.i_episode = 0
    mq = setup_mq_client
    agent.choose_action()
    agent.write_agent_step()

    agent_info = json.loads(mq.get("agent_step"))
    assert agent_info["episode"] == 0
    assert agent_info["step"] == 0


def test_episode_advancement(setup_mq_server, setup_mq_client):
    agent = initialize_agent()
    mq = setup_mq_client
    time.sleep(_pause)
    mq.put("control", "truncated")
    world_step_zero(mq)

    p_agent = mp.Process(target=agent.run)
    p_agent.start()
    time.sleep(_pause)

    i_episode = None
    for i in range(_max_retries):
        mq.put("control", "truncated")
        world_step_zero(mq)
        time.sleep(_long_pause)
        msg_str = mq.get("agent_step")
        debug_msg_str = mq.get("debug")
        print(f"retry {i} msg: {msg_str}")
        print(f"retry {i} debug: {debug_msg_str}")
        if msg_str == "":
            continue
        else:
            agent_info = json.loads(msg_str)
            print(f" agent episode {agent_info['episode']}")
            i_episode = agent_info["episode"]
            break

    assert i_episode == 1

    agent_exit = p_agent.join(_timeout)

    if p_agent.is_alive():
        p_agent.kill()
        time.sleep(_long_pause)
        p_agent.close()


'''
# test_termination

def test_reset_through_message_q(setup_mq_server, setup_mq_client):
    agent = initialize_agent()
    mq = setup_mq_client

    # time.sleep(_pause)
    world_step_zero(mq)
    # agent.run()
    time.sleep(_pause)
    agent_info = mq.get("agent_step")
    assert agent_info["episode"] == 0

    time.sleep(_long_pause)
    mq.put("control", "truncated")
    time.sleep(_pause)
    world_step_zero(mq)
    time.sleep(_long_pause)
'''
