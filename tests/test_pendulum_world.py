import multiprocessing as mp
import time
from myrtle.config import LOG_DIRECTORY
from myrtle.worlds import pendulum


def initialize_new_world():
    sen_q = mp.Queue()
    act_q = mp.Queue()
    rep_q = mp.Queue()
    log_name = f"world_{int(time.time())}"
    world = pendulum.Pendulum(
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
