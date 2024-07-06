from myrtle import bench
from myrtle.agents.q_learning_curiosity import QLearningCuriosity  # noqa: F401
from myrtle.worlds.pendulum_discrete import PendulumDiscrete  # noqa: F401


def run_demo():
    print(
        """
    Demo of Myrtle running Q-Learning with curiosity-driven exploration,
    learning to invert a pendulum--to balance it upside down.
    In the display below, this will look like hovering near the -180- mark.

    This demo runs for a thousand episodes of a thousand steps--
    about 3 days if you let it run to completion.
    That's just enough time for it to settle in to good (close to optimal) behavior.

    A random agent will score an average reward of about 0.4.
    A perfect score is close to 2.0.

    """
    )

    bench.run(
        QLearningCuriosity,
        PendulumDiscrete,
        agent_args={
            "discount_factor": 0.2,
            "learning_rate": 0.1,
        },
        world_args={
            "n_time_steps": int(1e3),
            "n_episodes": int(1e3),
            "speedup": 1,
        },
    )


if __name__ == "__main__":
    run_demo()
