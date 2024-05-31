import multiprocessing as mp
import time
import numpy as np
from myrtle.config import LOG_DIRECTORY
from myrtle.worlds import stationary_bandit

np.random.seed(42)
PAUSE = 0.01  # seconds
TIMEOUT = 1  # seconds


def initialize_new_world():
    sen_q = mp.Queue()
    act_q = mp.Queue()
    rep_q = mp.Queue()
    log_name = f"world_{int(time.time())}"
    world = stationary_bandit.StationaryBandit(
        sensor_q=sen_q,
        action_q=act_q,
        report_q=rep_q,
        log_name=log_name,
        log_dir=LOG_DIRECTORY,
    )

    return world, sen_q, act_q, rep_q, log_name


def test_initialization():
    world, sen_q, act_q, rep_q, log_name = initialize_new_world()
    assert world.sensor_q is sen_q
    assert world.action_q is act_q
    assert world.report_q is rep_q


def test_action_sensor_qs():
    world, sen_q, act_q, rep_q, log_name = initialize_new_world()
    p_world = mp.Process(target=world.run)

    p_world.start()
    time.sleep(PAUSE)
    act_q.put({"actions": np.array([0, 0, 1, 0, 0])})

    # Get the return message
    msg = sen_q.get(True, TIMEOUT)
    sensors = msg["sensors"]
    rewards = msg["rewards"]

    assert sensors.size == 0
    assert len(rewards) == 5
    assert rewards[0] == 0
    assert rewards[4] == 0

    p_world.kill()
    time.sleep(PAUSE)
    p_world.close()
