import multiprocessing as mp

# spawn is the default method on macOS,
# starting in Python 3.14 it will be the default in Linux too.
try:
    mp.set_start_method("spawn")
except RuntimeError:
    # Will throw an error if the start method has alraedy been set.
    pass

from importlib.metadata import version
import json
import sqlite3
from threading import Thread
import time

import dsmq.client
import dsmq.server
from myrtle.agents import base_agent
from myrtle.config import (
    log_directory,
    monitor_host,
    monitor_port,
    mq_host,
    mq_port,
)
from myrtle.monitors import server as monitor_server
from myrtle.worlds import base_world
from pacemaker.pacemaker import Pacemaker
from sqlogging import logging

_db_name_default = "bench"
_logging_frequency = 200.0  # Hz
_health_check_frequency = 10.0  # Hz
_warmup_delay = 2.0  # seconds
_shutdown_timeout = 1.0  # seconds
_shutdown_wait = 0.1  # seconds
_logging_retry_wait = 1 / _logging_frequency  # seconds


def run(
    Agent,
    World,
    log_to_db=True,
    logging_db_name=_db_name_default,
    timeout=None,
    agent_args={},
    world_args={},
    verbose=False,
):
    """
    log_to_db (bool)
    If True, log_to_db the results of this run in the results database.

    logging_db_name (str)
    A filename or path + filename to the database
    where the benchmark results are collected.

    timeout (int or None)
    How long in seconds the world and agent are allowed to run
    If None, then there is no timeout.

    """
    print(f"""

    Myrtle workbench version {version("myrtle")}
      World: {World.name}
      Agent: {Agent.name}

    Cancel run:                CTRL-c twice""")
    if log_to_db:
        print(f"""
    Check learning progress:   uv run reward_report {logging_db_name}

    Check timing:              uv run timing_report {logging_db_name}

      Reports are saved in the {log_directory} directory.
      A full history of the run is stored in {logging_db_name}.db """)

    control_pacemaker = Pacemaker(_health_check_frequency)

    print(f"""
    Watch learning progress:   http://{monitor_host}:{monitor_port}/bench.html""")

    # Kick off the message queue process
    p_mq_server = mp.Process(
        target=dsmq.server.serve, args=(mq_host, mq_port, _db_name_default, verbose)
    )
    p_mq_server.start()

    # Kick off the web server that shares monitoring pages
    p_monitor = mp.Process(target=monitor_server.serve)
    p_monitor.start()

    time.sleep(_warmup_delay)

    # Queues are the dedicated channels for agent and world to communicate
    # with each other, forming a tighter, faster, and more predictable loop
    # than the dsmq message queue.
    q_action = mp.Queue()
    q_reward = mp.Queue()
    q_sensor = mp.Queue()
    q_args = {
        "q_action": q_action,
        "q_reward": q_reward,
        "q_sensor": q_sensor,
    }

    world_args = world_args | q_args
    agent_args = agent_args | q_args

    world = World(**world_args)
    n_sensors = world.n_sensors
    n_actions = world.n_actions
    try:
        n_rewards = world.n_rewards
    except AttributeError:
        n_rewards = 1

    agent = Agent(
        n_sensors=n_sensors,
        n_actions=n_actions,
        n_rewards=n_rewards,
        **agent_args,
    )

    # Start up the logging thread, if it's called for.
    if log_to_db:
        t_logging = Thread(
            target=_reward_logging, args=(logging_db_name, agent, world, verbose)
        )
        t_logging.start()

    p_agent = mp.Process(target=agent.run)
    p_world = mp.Process(target=world.run)

    p_agent.start()
    p_world.start()

    # Keep the workbench alive until it's time to close it down.
    # Monitor a "control" topic for a signal to stop everything.
    mq_control_client = dsmq.client.connect(mq_host, mq_port)
    run_start_time = time.time()
    while True:
        control_pacemaker.beat()

        # Check whether a shutdown message has been sent.
        # Assume that there will not be high volume on the "control" topic
        # and just check this once.
        msg = mq_control_client.get("control")
        if msg is None:
            if verbose:
                print("dsmq server connection terminated unexpectedly.")
                print("Shutting it all down.")
            break

        try:
            if msg in ["terminated", "shutdown"]:
                if verbose:
                    print("==== workbench run terminated by another process ====")
                break
        except KeyError:
            pass

        if timeout is not None and time.time() - run_start_time > timeout:
            mq_control_client.put("control", "terminated")
            if verbose:
                print(f"==== workbench run timed out at {timeout} sec ====")
            break

        # TODO
        # Put heartbeat health checks for agent and world here.

    exitcode = 0
    if log_to_db:
        t_logging.join(_shutdown_timeout)
        if t_logging.is_alive():
            if verbose:
                print("    logging didn't shutdown cleanly")
            exitcode = 1

    monitor_server.shutdown()
    p_agent.join(_shutdown_timeout)
    p_world.join(_shutdown_timeout)

    time.sleep(_shutdown_wait)

    # Clean up any processes that might accidentally be still running.
    if p_monitor.is_alive():
        if verbose:
            print("    monitor webserver didn't shutdown cleanly")
        exitcode = 1
        p_monitor.kill()

    if p_world.is_alive():
        if verbose:
            print("    Doing a hard shutdown on world")
        exitcode = 1
        p_world.kill()

    if p_agent.is_alive():
        if verbose:
            print("    Doing a hard shutdown on agent")
        exitcode = 1
        p_agent.kill()

    mq_control_client.shutdown_server()
    mq_control_client.close()

    # If there are external connections to the mq server, like one of the
    # monitors, they won't allow it to shutdown gently.
    # When that happens, do this hard shutdiwn instead.
    # It's still considered healthy behavior and gives and exitcode of 0.
    if p_mq_server.is_alive():
        if verbose:
            print("    Doing a hard shutdown on mq server")
        p_mq_server.kill()

    return exitcode


def _reward_logging(dbname, agent, world, verbose):
    # Spin up the sqlite database where results are stored.
    # If a logger already exists, use it.
    try:
        logger = logging.open_logger(
            name=dbname,
            dir_name=log_directory,
            level="info",
        )
    except (sqlite3.OperationalError, RuntimeError):
        # If necessary, create a new logger.
        logger = logging.create_logger(
            name=dbname,
            dir_name=log_directory,
            columns=[
                "process",
                "reward",
                "step",
                "episode",
                "ts_recv",
                "ts_send",
            ],
        )
    logging_pacemaker = Pacemaker(_logging_frequency)

    mq_logging_client = dsmq.client.connect(mq_host, mq_port)
    while True:
        logging_pacemaker.beat()

        # Check whether a shutdown message has been sent.
        # Assume that there will not be high volume on the "control" topic
        # and just check this once.
        msg = mq_logging_client.get("control")
        if msg is None:
            if verbose:
                print("dsmq server connection terminated unexpectedly.")
                print("Shutting it all down.")
            break

        try:
            if msg in ["terminated", "shutdown"]:
                break
        except KeyError:
            pass

        # Check whether there is new world step and reward value reported.
        msg_str = mq_logging_client.get("world_step")
        if msg_str is None:
            if verbose:
                print("dsmq server connection terminated unexpectedly.")
            break
        if msg_str == "":
            continue
        msg = json.loads(msg_str)

        reward = 0.0
        try:
            for reward_channel in msg["rewards"]:
                if reward_channel is not None:
                    reward += reward_channel
        except KeyError:
            # Rewards not yet populated.
            pass

        log_data = {
            "process": "world",
            "reward": reward,
            "step": msg["loop_step"],
            "episode": msg["episode"],
            "ts_recv": msg["ts_recv"],
            "ts_send": msg["ts_send"],
        }
        try:
            logger.info(log_data)
        except sqlite3.OperationalError:
            # Retry once if database is locked
            time.sleep(_logging_retry_wait)
            logger.info(log_data)

        # Check whether there is new agent step reported.
        msg_str = mq_logging_client.get("agent_step")
        if msg_str is None:
            if verbose:
                print("dsmq server connection terminated unexpectedly.")
            break
        if msg_str == "":
            continue
        msg = json.loads(msg_str)

        log_data = {
            "process": "agent",
            "step": msg["step"],
            "episode": msg["episode"],
            "ts_recv": msg["ts_recv"],
            "ts_send": msg["ts_send"],
        }
        logger.info(log_data)

    # Gracefully close down logger and mq_client
    mq_logging_client.close()
    logger.close()


if __name__ == "__main__":
    exitcode = run(base_agent.BaseAgent, base_world.BaseWorld)
