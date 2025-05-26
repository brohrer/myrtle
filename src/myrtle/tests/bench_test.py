from multiprocessing import Process
import os
import time
import dsmq.client
from sqlogging import logging
from myrtle import bench
from myrtle.agents import base_agent
from myrtle.config import log_directory, mq_host, mq_port
from myrtle.worlds import base_world

_bench_run_timeout = 5.0  # seconds
_startup_delay = 5.0  # seconds
_test_db_name = f"temp_bench_test_{int(time.time())}"


def db_cleanup():
    db_filename = f"{_test_db_name}.db"
    db_path = os.path.join(log_directory, db_filename)
    os.remove(db_path)


def test_run_a():
    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=False,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
        verbose=True,
    )
    assert exitcode == 0


def test_run():
    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=False,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
    )
    assert exitcode == 0


def test_timeout():
    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=False,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5000,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
    )
    assert exitcode == 0


def test_logging():
    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=True,
        logging_db_name=_test_db_name,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
    )
    assert exitcode == 0

    db_cleanup()


def test_result_logging():
    exitcode = bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=True,
        logging_db_name=_test_db_name,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
    )
    assert exitcode == 0

    logger = logging.open_logger(
        name=_test_db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT step
        FROM {_test_db_name}
        ORDER BY ts_send DESC
        LIMIT 1
    """
    )
    assert result[0][0] == 4

    result = logger.query(
        f"""
        SELECT AVG(reward)
        FROM {_test_db_name}
        GROUP BY episode
        ORDER BY episode DESC
        LIMIT 1
    """
    )
    print(f"Average reward: {result[0][0]}")
    assert result[0][0] > -2 and result[0][0] < 2

    db_cleanup()


def test_multiple_runs():
    bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=True,
        logging_db_name=_test_db_name,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
    )

    bench.run(
        base_agent.BaseAgent,
        base_world.BaseWorld,
        log_to_db=True,
        logging_db_name=_test_db_name,
        timeout=_bench_run_timeout,
        world_args={
            "n_loop_steps": 5,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
    )

    logger = logging.open_logger(
        name=_test_db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT COUNT(DISTINCT episode)
        FROM {_test_db_name}
    """
    )
    print(f"Number of workbench runs: {result[0][0]}")
    assert result[0][0] == 2

    db_cleanup()


def test_controlled_shutdown():
    run_args = (
        base_agent.BaseAgent,
        base_world.BaseWorld,
    )
    run_kwargs = {
        "log_to_db": True,
        "logging_db_name": _test_db_name,
        "world_args": {
            "n_loop_steps": 50000,
            "n_episodes": 2,
            "loop_steps_per_second": 20,
        },
        "verbose": True,
    }
    p_bench_run = Process(target=bench.run, args=run_args, kwargs=run_kwargs)
    p_bench_run.start()
    time.sleep(_startup_delay)

    mq = dsmq.client.connect(mq_host, mq_port)
    mq.put("control", "terminated")
    mq.close()

    p_bench_run.join(_bench_run_timeout)

    assert not p_bench_run.is_alive()
    assert p_bench_run.exitcode == 0

    db_cleanup()


if __name__ == "__main__":
    test_run_a()
    # test_controlled_shutdown()
