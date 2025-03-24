import time
from myrtle import bench
from myrtle.agents.q_learning_curiosity import QLearningCuriosity
from myrtle.config import log_directory
from myrtle.worlds.contextual_bandit import ContextualBandit
from sqlogging import logging


def run_demo():
    print(
        """
    Demo of Myrtle running Q-Learning with curiosity-driven exploration,
    learning a contextual bandit--a multi-armed bandit where sensor values
    indicate which one has the highest payout.

    This demo runs for 20,000 steps,
    at 100 steps per second--a little over 3 minutes.
    That's just enough time for it
    to settle in to good (close to optimal) behavior.

    """
    )

    db_name = f"demo_db_{int(time.time())}"
    bench.run(
        QLearningCuriosity,
        ContextualBandit,
        log_to_db=True,
        logging_db_name=db_name,
        agent_args={
            "curiosity_scale": 100.0,
            "discount_factor": 0.0,
            "learning_rate": 0.01,
        },
        world_args={
            "n_loop_steps": 20000,
            "loop_steps_per_second": 100,
            "verbose": True,
        },
    )

    logger = logging.open_logger(
        name=db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT AVG(reward)
        FROM {db_name}
    """
    )
    average_reward = result[0][0]
    print(f"""

    Average reward over the run was about {int(average_reward)}.
    """)

    print(
        """
    A random agent will score an average reward in the neighborhood of 65.
    A perfect score is closer to 110.

    Dig in to README.md for usage, examples, and API documentation.

    """
    )
    print()


if __name__ == "__main__":
    run_demo()
