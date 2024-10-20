import multiprocessing as mp

mp.set_start_method("fork")

import time  # noqa: E402
from myrtle import bench_dash  # noqa: E402
from myrtle.agents import base_agent  # noqa: E402
from myrtle.worlds import base_world  # noqa: E402
from sqlogging import logging  # noqa: E402

PAUSE = 0.01  # seconds
DB_NAME_DEFAULT = "bench"

WINDOWS_LEFT_PIXEL = 0
WINDOWS_TOP_PIXEL = 0
WINDOWS_WIDTH_PIXELS = 1800
WINDOWS_HEIGHT_PIXELS = 1000
WINDOW_TITLE_HEIGHT = 100
BENCH_HEIGHT_FRACTION = 0.35


def run(
    Agent,
    World,
    timeout=None,
    record=True,
    db_name=DB_NAME_DEFAULT,
    # (x, y, width, height)
    window_pixels=(
        WINDOWS_LEFT_PIXEL,
        WINDOWS_TOP_PIXEL,
        WINDOWS_WIDTH_PIXELS,
        WINDOWS_HEIGHT_PIXELS,
    ),
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

    window_pixels (tuple(int))
    The location of the windows for myrtle's dashboard constellation.
        * x-pixel of the left edge of the window
        * y-pixel of the top edge of the window
            (counting down from the top of the screen)
        * width of the window in pixels
        * height of the window in pixels
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

    x, y, width, height = window_pixels
    bench_height = int(height * BENCH_HEIGHT_FRACTION) # - WINDOW_TITLE_HEIGHT
    half_width = int(width / 2)
    bench_window = (
        x,
        y,
        half_width,
        bench_height
    )
    world_window = (
        x,
        y + bench_height + WINDOW_TITLE_HEIGHT,
        half_width,
        height - bench_height - WINDOW_TITLE_HEIGHT
    )
    agent_window = (
        x + half_width,
        y,
        half_width,
        height - WINDOW_TITLE_HEIGHT
    )

    sensor_q = mp.Queue()
    action_q = mp.Queue()
    dash_q = mp.Queue()
    report_q = mp.Queue()

    world = World(
        sensor_q=sensor_q,
        action_q=action_q,
        report_q=report_q,
        window_pixels=world_window,
        **world_args
    )
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

    p_dash = mp.Process(target=bench_dash.run, args=(dash_q, bench_window))
    p_dash.start()

    total_reward = 0.0
    total_steps = 0
    while True:
        # This is a blocking call.
        # It serves as the pacemaker for this loop.
        msg = report_q.get()
        try:
            if msg["terminated"]:
                break
        except KeyError:
            pass
        # Empty the queue in case is gets behind.
        # But check for termination each time.
        while not report_q.empty():
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
        dash_q.put((reward, step, episode))

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
    print()
    if episode > 1:
        print(
            f"Lifetime average reward across {episode + 1} episodes"
            + f" of {step} steps each"
        )
        print(f"for {agent.name} on {world.name}: {avg_reward}")
    else:
        print(
            f"    Lifetime average reward for {agent.name}"
            + f" on {world.name}: {avg_reward}"
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
