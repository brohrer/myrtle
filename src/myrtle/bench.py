import multiprocessing as mp

mp.set_start_method("fork")

import time  # noqa: E402
from myrtle.agents import base_agent  # noqa: E402
from myrtle.worlds import base_world  # noqa: E402
from sqlogging import logging  # noqa: E402

PAUSE = 0.01  # seconds
DB_NAME_DEFAULT = "bench"


def run(
    Agent,
    World,
    timeout=None,
    record=True,
    db_name=DB_NAME_DEFAULT,
    agent_args={},
    world_args={},
):
    """
    timeout (int or None)
    How long in seconds the world and agent are allowed to run
    If None, then there is no timeout.

    record (bool)
    If True, record the results of this run in the results database.

    db_name (str)
    A filename or path + filename to the database where the benchmark results are
    collected.
    """
    run_timestamp = time.time()

    # Spin up the sqlite database where results are stored
    if record:
        # If a logger already exists, use it.
        try:
            logger = logging.open_logger(name=db_name)
        except RuntimeError:
            # If necessary, create a new logger.
            logger = logging.create_logger(
                name=db_name,
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

    sensor_q = mp.Queue()
    action_q = mp.Queue()
    report_q = mp.Queue()

    world = World(sensor_q=sensor_q, action_q=action_q, report_q=report_q, **world_args)
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
        sensor_q=sensor_q,
        action_q=action_q,
        **agent_args,
    )

    p_agent = mp.Process(target=agent.run)
    p_world = mp.Process(target=world.run)

    p_agent.start()
    p_world.start()

    total_reward = 0.0
    total_steps = 0
    while True:
        msg = report_q.get()
        try:
            if msg["terminated"]:
                break
        except KeyError:
            pass

        reward = 0.0
        for reward_channel in msg["rewards"]:
            if reward_channel is not None:
                reward += reward_channel

        total_reward += reward
        total_steps += 1

        step = msg["step"]
        episode = msg["episode"]
        if record:
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

    # Put a human-readable report in the console
    avg_reward = total_reward / total_steps
    print(
        f"Avg reward for {agent.name} on {world.name},"
        + f" across {episode + 1} episodes"
        + f" with {step} steps each: {avg_reward}"
    )

    world_exit = p_world.join()
    agent_exit = p_agent.join()

    if world_exit is None and agent_exit is None:
        exitcode = 0
    else:
        exitcode = 1

    # Clean up any processes that might accidentally be still running.
    if p_world.is_alive():
        p_world.kill()
        time.sleep(PAUSE)
        p_world.close()

    if p_agent.is_alive():
        p_agent.kill()
        time.sleep(PAUSE)
        p_agent.close()

    return exitcode


if __name__ == "__main__":
    exitcode = run(base_agent.BaseAgent, base_world.BaseWorld)
    print(exitcode)
