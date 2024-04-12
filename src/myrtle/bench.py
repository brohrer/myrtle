import multiprocessing as mp
mp.set_start_method("fork")

def run(Agent, World):
    world = World()
    n_sensors = world.n_sensors
    n_actions = world.n_actions
    try:
        n_rewards = world.n_rewards
    catch:
        n_rewards = 1

    agent = Agent(
        n_sensors=n_sensors,
        n_actions=n_actions,
        n_rewards=n_rewards,
    )

    sensor_q = mp.Queue()
    action_q = mp.Queue()
    kwdict = {"sensor_q": sensor_q, "action_q": action_q}
    p_agent = mp.Process(target=Agent.run, kwargs=kwdict)
    p_world = mp.Process(target=World.run, kwargs=kwdict)

    p_agent.start()
    p_world.start()
