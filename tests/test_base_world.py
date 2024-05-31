import multiprocessing as mp
import signal
import time
import numpy as np
from sqlogging import logging
from myrtle.worlds import base_world
from myrtle.config import LOG_DIRECTORY

np.random.seed(42)
PAUSE = 0.01  # seconds
LONG_PAUSE = 0.3  # seconds
TIMEOUT = 1  # seconds


def initialize_new_base_world():
    sen_q = mp.Queue()
    act_q = mp.Queue()
    rep_q = mp.Queue()
    log_name = f"world_{int(time.time())}"
    world = base_world.BaseWorld(
        sensor_q=sen_q,
        action_q=act_q,
        report_q=rep_q,
        log_name=log_name,
        log_dir=LOG_DIRECTORY,
    )

    return world, sen_q, act_q, rep_q, log_name


def test_initialization():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    assert world.sensor_q is sen_q
    assert world.action_q is act_q
    assert world.report_q is rep_q


def test_sensor_reward_generation():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    world.reset()
    world.actions = np.array([0, 0, 1, 0, 0])
    world.step()
    assert world.sensors[1] == 0.0
    assert world.sensors[2] == 1.0
    assert world.sensors[7] == 0.5
    assert world.sensors[9] == -0.3
    assert world.rewards[0] == 0.2
    assert world.rewards[2] is None


def test_creation_and_natural_termination():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    p_world = mp.Process(target=world.run)

    p_world.start()
    time.sleep(PAUSE)
    assert p_world.is_alive() is True
    assert p_world.exitcode is None

    p_world.join()
    assert p_world.is_alive() is False
    assert p_world.exitcode == 0


def test_early_termination():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    p_world = mp.Process(target=world.run)

    p_world.start()
    time.sleep(PAUSE)
    p_world.kill()
    time.sleep(PAUSE)
    assert p_world.is_alive() is False
    assert p_world.exitcode == -signal.SIGKILL

    p_world.close()


def test_action_sensor_qs():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    p_world = mp.Process(target=world.run)

    p_world.start()
    time.sleep(PAUSE)
    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})

    # Get the return message
    msg = sen_q.get(True, TIMEOUT)
    sensors = msg["sensors"]
    rewards = msg["rewards"]

    assert sensors[1] == 0.0
    assert sensors[2] == 1.0
    assert sensors[7] == 0.5
    assert sensors[9] == -0.3
    assert rewards[0] == 0.2
    assert rewards[2] is None

    p_world.kill()
    time.sleep(PAUSE)
    p_world.close()


def test_action_sensor_logging():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    p_world = mp.Process(target=world.run)

    p_world.start()

    logger = logging.open_logger(
        name=log_name,
        dir_name=LOG_DIRECTORY,
    )

    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})
    time.sleep(LONG_PAUSE)

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
    time.sleep(PAUSE)
    p_world.close()


def test_termination_truncation():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
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


def test_report_q():
    world, sen_q, act_q, rep_q, log_name = initialize_new_base_world()
    p_world = mp.Process(target=world.run)

    p_world.start()
    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})

    # Get the return message
    msg = rep_q.get(True, TIMEOUT)
    i_step = msg["step"]
    i_episode = msg["episode"]
    rewards = msg["rewards"]

    assert rewards[0] == 0.2
    assert rewards[2] is None
    assert i_step == 1
    assert i_episode == 0

    p_world.kill()
    time.sleep(PAUSE)
    p_world.close()
