import multiprocessing as mp
import signal
import time
import numpy as np
from sqlogging import logging
from myrtle.agents import base_agent
from myrtle.config import LOG_DIRECTORY

np.random.seed(42)
PAUSE = .01  # seconds
TIMEOUT = 1  # seconds


def initialize_new_base_world():
    sen_q = mp.Queue()
    act_q = mp.Queue()
    log_name = f"agent_{int(time.time())}"
    agent = base_agent.BaseAgent(
        sensor_q=sen_q,
        action_q=act_q,
        log_name=log_name,
        log_dir=LOG_DIRECTORY,
    )

    return world, sen_q, act_q, log_name


def test_initialization():
    world, sen_q, act_q, log_name = initialize_new_base_world()
    assert world.sensor_q is sen_q
    assert world.action_q is act_q



