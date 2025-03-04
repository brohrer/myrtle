from importlib.metadata import version
import json
import multiprocessing as mp
import os
import sqlite3
from threading import Thread
import time

import dsmq.client
import dsmq.server
import myrtle
from myrtle.agents import base_agent
from myrtle.config import log_directory, monitor_host, monitor_port, mq_host, mq_port
from myrtle.monitors import server as monitor_server
from myrtle.worlds import base_world
from pacemaker.pacemaker import Pacemaker
from sqlogging import logging

_db_name_default = "bench"
_logging_frequency = 100  # Hz
_health_check_frequency = 10.0  # Hz
_warmup_delay = 2.0  # seconds
_shutdown_timeout = 1.0  # seconds


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
    print()
    print(f"Myrtle workbench version {version('myrtle')}")
    print(f"  World: {World.name}")
    print(f"  Agent: {Agent.name}")
    print()
    control_pacemaker = Pacemaker(_health_check_frequency)

    # If a prior run crashed, it can leave a file that slows down
    # dsmq by encouraging to read/write to disk.
    try:
        os.remove("file::memory:?cache=shared")
    except FileNotFoundError:
        pass

    try:
        os.remove("file::memory:?cache=shared-journal")
    except FileNotFoundError:
        pass

    print("Follow this run at")
    print(f"http://{monitor_host}:{monitor_port}/bench.html")

    # Kick off the message queue process
    p_mq_server = mp.Process(
        target=dsmq.server.serve,
        args=(mq_host, mq_port),
    )
    p_mq_server.start()

    # Kick off the web server that shares monitoring pages
    p_monitor = mp.Process(target=monitor_server.serve)
    p_monitor.start()

    time.sleep(_warmup_delay)

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
    mq_client = dsmq.client.connect(mq_host, mq_port)
    run_start_time = time.time()
    while True:
        control_pacemaker.beat()

        # Check whether a shutdown message has been sent.
        # Assume that there will not be high volume on the "control" topic
        # and just check this once.
        msg = mq_client.get("control")
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
            mq_client.put("control", "terminated")
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

    # Clean up any processes that might accidentally be still running.
    # if t_monitor.is_alive():
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

    # Shutdown the mq server last
    mq_client.shutdown_server()
    mq_client.close()

    # If there are external connections to the mq server, like one of the
    # monitors, they won't allow it to shutdown gently.
    # When that happens, do this hard shutdiwn instead.
    # It's still considered healthy behavior and gives and exitcode of 0.
    if p_mq_server.is_alive():
        if verbose:
            print("    Doing a hard shutdown on mq server")
        p_mq_server.kill()

    # If clients crashed or haven't closed down cleanly,
    # they  can leave a file that slows down
    # dsmq by encouraging to read/write to disk.
    try:
        os.remove("file::memory:?cache=shared")
    except FileNotFoundError:
        pass

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
                "reward",
                "step",
                "step_timestamp",
                "episode",
                "run_timestamp",
                "agentname",
                "worldname",
            ],
        )
    run_timestamp = time.time()
    logging_pacemaker = Pacemaker(_logging_frequency)

    logging_mq_client = dsmq.client.connect(mq_host, mq_port)
    while True:
        logging_pacemaker.beat()

        # Check whether a shutdown message has been sent.
        # Assume that there will not be high volume on the "control" topic
        # and just check this once.
        msg = logging_mq_client.get("control")
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

        # Check whether there is new reward value reported.
        msg_str = logging_mq_client.get("world_step")
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

        step = msg["loop_step"]
        episode = msg["episode"]

        log_data = {
            "reward": reward,
            "step": step,
            "step_timestamp": time.time(),
            "episode": episode,
            "run_timestamp": run_timestamp,
            "agentname": agent.name,
            "worldname": world.name,
        }
        logger.info(log_data)

    # Gracefully close down logger and mq_client
    logging_mq_client.close()
    logger.close()


if __name__ == "__main__":
    exitcode = run(base_agent.BaseAgent, base_world.BaseWorld)
