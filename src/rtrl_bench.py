import multiprocessing as mp
mp.set_start_method("fork")

def run(Agent, World):
    world = World()
    agent = Agent(
        n_sensors=world.n_sensors,
        n_actions=world.n_actions,
        n_rewards=world.n_rewards)

    q = mp.Queue()
    p_agent = mp.Process(target=Agent.run, args=(q,))
    p_world = mp.Process(target=World.run, args=(q,))

    p_agent.start()
    p_world.start()
