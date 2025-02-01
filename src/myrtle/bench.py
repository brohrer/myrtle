import dsmq.client
import dsmq.server
import json
import multiprocessing as mp
import sqlite3
from threading import Thread
import time
# mp.set_start_method("fork")

from myrtle import config
from myrtle.agents import base_agent
from myrtle.worlds import base_world
from pacemaker.pacemaker import Pacemaker
from sqlogging import logging

_db_name_default = "bench"
_logging_frequency = 100  # Hz
_health_check_frequency = 10.0  # Hz
_mq_server_setup_delay = 1.0  # seconds
_shutdown_delay = 4.0  # seconds
_kill_delay = 0.01  # seconds


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
    timeout (int or None)
    How long in seconds the world and agent are allowed to run
    If None, then there is no timeout.

    log_to_db (bool)
    If True, log_to_db the results of this run in the results database.

    logging_db_name (str)
    A filename or path + filename to the database where the benchmark results are
    collected.
    """
    control_pacemaker = Pacemaker(_health_check_frequency)

    # Kick off the message queue process
    p_mq_server = mp.Process(
        target=dsmq.server.serve,
        args=(config.MQ_HOST, config.MQ_PORT),
    )
    p_mq_server.start()
    time.sleep(_mq_server_setup_delay)

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
        t_logging = Thread(target=_reward_logging, args=(logging_db_name, agent, world))
        t_logging.start()

    p_agent = mp.Process(target=agent.run)
    p_world = mp.Process(target=world.run)

    p_agent.start()
    p_world.start()

    # Keep the workbench alive until it's time to close it down.
    # Monitor a "control" topic for a signal to stop everything.
    mq_client = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)
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
        t_logging.join(_shutdown_delay)
        if t_logging.is_alive():
            if verbose:
                print("    logging didn't shutdown cleanly")
            exitcode = 1

    p_agent.join(_shutdown_delay)
    p_world.join(_shutdown_delay)

    # Clean up any processes that might accidentally be still running.
    if p_world.is_alive():
        if verbose:
            print("    Doing a hard shutdown on world")
        exitcode = 1
        p_world.kill()
        time.sleep(_kill_delay)
        p_world.close()

    if p_agent.is_alive():
        if verbose:
            print("    Doing a hard shutdown on agent")
        exitcode = 1
        p_agent.kill()
        time.sleep(_kill_delay)
        p_agent.close()

    # Shutdown the mq server last
    mq_client.shutdown_server()
    mq_client.close()

    return exitcode


def _reward_logging(dbname, agent, world):
    # Spin up the sqlite database where results are stored.
    # If a logger already exists, use it.
    try:
        logger = logging.open_logger(
            name=dbname,
            dir_name=config.LOG_DIRECTORY,
            level="info",
        )
    except (sqlite3.OperationalError, RuntimeError):
        # If necessary, create a new logger.
        logger = logging.create_logger(
            name=dbname,
            dir_name=config.LOG_DIRECTORY,
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

    logging_mq_client = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)
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
