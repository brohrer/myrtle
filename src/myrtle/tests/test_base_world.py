import multiprocessing as mp
import pytest
import signal
import time
import numpy as np
import dsmq.server
from sqlogging import logging
from myrtle import config
from myrtle.agents import base_agent
from myrtle.worlds import base_world
from myrtle.config import LOG_DIRECTORY

np.random.seed(42)
# times in seconds
_pause = 0.01
_long_pause = 0.3
_v_long_pause = 1.0
_timeout = 1.0


@pytest.fixture
def setup_and_teardown():
    # Kick off the dsmq server in a separate process
    p_mq = mp.Process(target=dsmq.server.serve, args=(config.MQ_HOST, config.MQ_PORT))
    p_mq.start()
    time.sleep(_long_pause)
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)
    p_world.start()
    agent = base_agent.BaseAgent()
    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    yield agent, world

    # Shut down the dsmq process
    p_agent.join(_timeout)
    p_world.join(_timeout)
    p_mq.join(_timeout)

    time.sleep(_long_pause)

    if p_agent.is_alive():
        p_agent.kill()
        time.sleep(_pause)
        p_agent.close()

    if p_world.is_alive():
        p_world.kill()
        time.sleep(_pause)
        p_world.close()

    if p_mq.is_alive():
        p_mq.kill()
        time.sleep(_pause)
        p_mq.close()


def test_initialization(setup_and_teardown):
    world = base_world.BaseWorld()
    assert world.name == "Base world"
    assert world.n_sensors == 13
    assert world.n_actions == 5
    assert world.n_rewards == 3
    assert world.n_loop_steps == 100
    assert world.n_episodes == 1
    assert world.loop_steps_per_second == pytest.approx(10.0)
    assert world.world_steps_per_second == pytest.approx(10.0)
    assert world.pm.clock_period == pytest.approx(0.10)


def test_sensor_reward_generation(setup_and_teardown):
    world = base_world.BaseWorld()
    world.reset()
    time.sleep(_pause)
    world.actions = np.array([0, 0, 1, 0, 0])
    world.i_loop_step = 0

    world.step_world()
    assert world.i_action == 2

    world.sense()
    assert world.sensors[1] == 0.0
    assert world.sensors[9] == -0.3
    assert world.rewards[0] == 0.2
    assert world.rewards[2] is None


# Test connection to dsmq
def test_creation_and_natural_termination(setup_and_teardown):
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)

    p_world.start()
    time.sleep(_pause)
    assert p_world.is_alive() is True
    assert p_world.exitcode is None

    p_world.join(_timeout)
    assert p_world.is_alive() is False
    assert p_world.exitcode == 0


'''
def test_early_termination(setup_and_teardown):
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)

    p_world.start()
    time.sleep(_pause)
    p_world.kill()
    time.sleep(_pause)
    assert p_world.is_alive() is False
    assert p_world.exitcode == -signal.SIGKILL

    p_world.close()


def test_action_sensor_qs(setup_and_teardown):
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)

    p_world.start()
    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})
    time.sleep(_pause)

    # Get the return message
    msg = sen_q.get(True, _timeout)
    sensors = msg["sensors"]
    rewards = msg["rewards"]

    # assert sensors is None
    assert sensors[1] == 0.0
    assert sensors[2] == 1.0
    assert sensors[7] == 0.5
    assert sensors[9] == -0.3
    assert rewards[0] == 0.2
    assert rewards[2] is None

    p_world.kill()
    time.sleep(_pause)
    p_world.close()


def test_action_sensor_logging(setup_and_teardown):
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)

    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})
    p_world.start()
    time.sleep(_pause)

    logger = logging.open_logger(
        name=log_name,
        dir_name=LOG_DIRECTORY,
    )

    time.sleep(_long_pause)

    def get_value(col):
        result = logger.query(
            f"""
            SELECT {col}
            FROM {log_name}
            ORDER BY timestamp ASC
            LIMIT 1
        """
        )
        return result[0][0]

    assert get_value("sen1") == 0.0
    assert get_value("sen2") == 1.0
    assert get_value("sen7") == 0.5
    assert get_value("sen9") == -0.3
    assert get_value("rew0") == 0.2
    assert get_value("rew2") is None

    p_world.kill()
    time.sleep(_pause)
    p_world.close()


def test_termination_truncation(setup_and_teardown):
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)

    p_world.start()

    p_world.join()

    termination_count = 0
    truncation_count = 0

    while not sen_q.empty():
        msg = sen_q.get()
        try:
            if msg["truncated"]:
                truncation_count += 1
        except KeyError:
            pass

        try:
            if msg["terminated"]:
                termination_count += 1
        except KeyError:
            pass

    assert truncation_count == 2
    assert termination_count == 1


def test_report_q(setup_and_teardown):
    world = base_world.BaseWorld()
    p_world = mp.Process(target=world.run)

    p_world.start()
    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})

    # Get the return message
    msg = rep_q.get(True, _timeout)
    i_step = msg["step"]
    i_episode = msg["episode"]
    rewards = msg["rewards"]

    assert rewards[0] == 0.2
    assert rewards[2] is None
    assert i_step == 1
    assert i_episode == 0

    p_world.kill()
    time.sleep(_pause)
    p_world.close()
'''
