import multiprocessing as mp
mp.set_start_method("fork")

def run(Agent, World, timeout=None):
    """
    timeout (int or None)
    How long in seconds the world and agent are allowed to run
    If None, then there is no timeout.
    """
    sensor_q = mp.Queue()
    action_q = mp.Queue()

    world = World(sensor_q=sensor_q, action_q=action_q)
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
    )

    p_agent = mp.Process(target=agent.run)
    p_world = mp.Process(target=world.run)

    p_agent.start()
    p_world.start()

    if timeout is not None:
        world_exit = p_world.join(timeout)
        agent_exit = p_agent.join(timeout)
        p_agent.kill()
        time.sleep(PAUSE)
        p_agent.close()
    else:
        world_exit = p_world.join()
        agent_exit = p_agent.join()
        p_agent.kill()
        time.sleep(PAUSE)
        p_agent.close()

    if world_exit is None or agent_exit is None:
        exitcode = 1
    elif world_exit == 0 and agent_exit == 0:
        exitcode = 0
    else: 
        exitcode = 1

    # return exitcode
    return world_exit
