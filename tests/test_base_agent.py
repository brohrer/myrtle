import multiprocessing as mp
import signal
import time
import numpy as np
from sqlogging import logging
from myrtle.agents import base_agent
from myrtle.config import LOG_DIRECTORY

np.random.seed(42)
PAUSE = 0.01  # seconds
TIMEOUT = 1  # seconds


def initialize_new_base_agent():
    n_sen = np.random.randint(3, 17)
    n_act = np.random.randint(2, 14)
    n_rew = np.random.randint(2, 5)
    sen_q = mp.Queue()
    act_q = mp.Queue()
    log_name = f"agent_{int(time.time())}"
    agent = base_agent.BaseAgent(
        n_sensors=n_sen,
        n_actions=n_act,
        n_rewards=n_rew,
        sensor_q=sen_q,
        action_q=act_q,
        log_name=log_name,
        log_dir=LOG_DIRECTORY,
    )

    return agent, n_sen, n_act, n_rew, sen_q, act_q, log_name


def test_initialization():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()

    assert agent.n_sensors == n_sen
    assert agent.n_actions == n_act
    assert agent.n_rewards == n_rew
    assert agent.sensor_q is sen_q
    assert agent.action_q is act_q


def test_action_generation():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()
    agent.step()

    # There should be just one nonzero action, and it should have a value of 1.
    assert agent.actions.size == n_act
    assert np.where(agent.actions > 0)[0].size == 1
    assert np.sum(agent.actions) == 1


def test_process_creation_and_destruction():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()
    p_agent = mp.Process(target=agent.run)
    assert p_agent.is_alive() is False

    p_agent.start()
    time.sleep(PAUSE)
    assert p_agent.is_alive() is True
    assert p_agent.exitcode is None

    p_agent.kill()
    time.sleep(PAUSE)
    assert p_agent.is_alive() is False
    assert p_agent.exitcode == -signal.SIGKILL

    p_agent.close()


def test_shutdown_through_sensor_q():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()
    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    msg = {"terminated": True}
    sen_q.put(msg)
    time.sleep(PAUSE)

    assert p_agent.is_alive() is False
    assert p_agent.exitcode == 0

    p_agent.close()


def test_reset():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()
    assert agent.i_step == 0

    agent.step()
    agent.i_step += 1

    agent.step()
    agent.i_step += 1

    agent.reset()
    assert agent.i_step == 0

    agent.logger.delete()


def test_reset_through_sensor_q():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()

    agent.step()
    agent.log_step()
    agent.i_step += 1

    agent.step()
    agent.log_step()
    agent.i_step += 1

    agent.step()
    agent.log_step()

    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    logger = logging.open_logger(
        name=log_name,
        dir_name=LOG_DIRECTORY,
    )
    result = logger.query(
        f"""
        SELECT i_step
        FROM {log_name}
        ORDER BY timestamp DESC
        LIMIT 1
    """
    )
    assert result[0][0] == 2

    msg = {"truncated": True}
    sen_q.put(msg)
    time.sleep(PAUSE)

    result = logger.query(
        f"""
        SELECT i_step
        FROM {log_name}
        ORDER BY timestamp DESC
        LIMIT 1
    """
    )
    assert result[0][0] == 0

    p_agent.kill()
    time.sleep(PAUSE)
    p_agent.close()


def test_reward_q():
    """
    Verify that the rewards passed into the base agent are received and stored.
    """
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()

    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    logger = logging.open_logger(
        name=log_name,
        dir_name=LOG_DIRECTORY,
    )

    sensors = np.ones(n_sen)
    rewards = np.ones(n_rew) * 789
    msg = {
        "sensors": sensors,
        "rewards": rewards,
    }
    sen_q.put(msg)
    time.sleep(PAUSE)

    for i_rew in range(n_rew):
        result = logger.query(
            f"""
            SELECT rew{i_rew}
            FROM {log_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        assert result[0][0] == 789

    rewards = np.ones(n_rew) * 543
    msg = {
        "sensors": sensors,
        "rewards": rewards,
    }
    sen_q.put(msg)
    time.sleep(PAUSE)

    for i_rew in range(n_rew):
        result = logger.query(
            f"""
            SELECT rew{i_rew}
            FROM {log_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        assert result[0][0] == 543

    p_agent.kill()
    time.sleep(PAUSE)
    p_agent.close()


def test_sensor_q():
    """
    Verify that the sensor values passed into the base agent are received and stored.
    """
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()

    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    logger = logging.open_logger(
        name=log_name,
        dir_name=LOG_DIRECTORY,
    )

    sensors = np.ones(n_sen) * 267
    rewards = np.ones(n_rew)
    msg = {
        "sensors": sensors,
        "rewards": rewards,
    }
    sen_q.put(msg)
    time.sleep(PAUSE)

    for i_sen in range(n_sen):
        result = logger.query(
            f"""
            SELECT sen{i_sen}
            FROM {log_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        assert result[0][0] == 267

    sensors = np.ones(n_sen) * 7389
    msg = {
        "sensors": sensors,
        "rewards": rewards,
    }
    sen_q.put(msg)
    time.sleep(PAUSE)

    for i_sen in range(n_sen):
        result = logger.query(
            f"""
            SELECT sen{i_sen}
            FROM {log_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        assert result[0][0] == 7389

    p_agent.kill()
    time.sleep(PAUSE)
    p_agent.close()


def test_actions():
    (
        agent,
        n_sen,
        n_act,
        n_rew,
        sen_q,
        act_q,
        log_name,
    ) = initialize_new_base_agent()

    p_agent = mp.Process(target=agent.run)
    p_agent.start()

    logger = logging.open_logger(
        name=log_name,
        dir_name=LOG_DIRECTORY,
    )

    sensors = np.ones(n_sen)
    rewards = np.ones(n_rew)
    msg = {
        "sensors": sensors,
        "rewards": rewards,
    }
    sen_q.put(msg)
    time.sleep(PAUSE)
    # Get the return message
    msg = act_q.get(True, TIMEOUT)
    actions = msg["actions"]

    # There should be just one nonzero action, and it should have a value of 1.
    assert actions.size == n_act
    assert np.where(actions > 0)[0].size == 1
    assert np.sum(actions) == 1

    for i_act in range(n_act):
        result = logger.query(
            f"""
            SELECT act{i_act}
            FROM {log_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        assert result[0][0] == actions[i_act]

    p_agent.kill()
    time.sleep(PAUSE)
    p_agent.close()
