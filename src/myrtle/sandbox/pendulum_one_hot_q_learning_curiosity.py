import os
import time
from sqlogging import logging
from myrtle import bench
from myrtle.config import log_directory
from myrtle.agents.q_learning_curiosity import QLearningCuriosity
from myrtle.worlds.pendulum_discrete import PendulumDiscrete
# from myrtle.worlds.pendulum_discrete_one_hot import PendulumDiscreteOneHot


# curiosity_scale = 1.0
curiosity_scale = 0.1
discount_factor = 0.5
learning_rate = 0.1

n_loop_steps = 1e10
n_episodes = 1
loops_per_second = 8
speedup = 8
verbose = False

db_name = f"q_curiosity_pendulum_{int(time.time())}"

# Try
#
# speedup = 8 , curiosity scale = 0.1
# self.action_scale = 4 * np.array(  speedup = 4
# self.action_scale = 2 * np.array(
# self.action_scale = 4 * np.array(  # was 8
# discount_factor: 0.5, learning_rate:0.1, loops_per_second: 8, speedup_8

def main():
    start_time = time.time()

    bench.run(
        QLearningCuriosity,
        PendulumDiscrete,
        # PendulumDiscreteOneHot,
        log_to_db=True,
        logging_db_name=db_name,
        world_args={
            "n_loop_steps": n_loop_steps,
            "n_episodes": n_episodes,
            "loop_steps_per_second": loops_per_second,
            "speedup": speedup,
            "verbose": verbose,
        },
        agent_args={
            "curiosity_scale": curiosity_scale,
            "discount_factor": discount_factor,
            "learning_rate": learning_rate,
        },
    )

    run_time = time.time() - start_time
    print()
    print(f"Ran in {int(run_time)} seconds")

    logger = logging.open_logger(
        name=db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT AVG(reward)
        FROM {db_name}
        GROUP BY episode
        ORDER BY episode DESC
    """
    )
    for i_episode in range(len(result)):
        print(f"Episode {i_episode} average reward: {result[i_episode][0]}")
    print()

    db_filename = f"{db_name}.db"
    db_path = os.path.join(log_directory, db_filename)
    os.remove(db_path)


if __name__ == "__main__":
    main()
