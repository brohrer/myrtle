import multiprocessing as mp
mp.set_start_method("fork")

def run(Agent, World):
    q = mp.Queue()
    p_agent = mp.Process(target=Agent.run, args=(q,))
    p_world = mp.Process(target=World.run, args=(q,))

    p_agent.start()
    p_world.start()
