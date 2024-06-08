from myrtle import bench
from myrtle.agents.q_learning_curiosity import QLearningCuriosity  # noqa: F401
from myrtle.worlds.contextual_bandit import ContextualBandit  # noqa: F401


def run_demo():
    print(
        """
    Demo of Myrtle running Q-Learning with curiosity-driven exploration,
    learning a contextual bandit--a multi-armed bandit where sensor values
    indicate which one has the highest payout.

    This demo runs for 20,000 steps, about 20 seconds.
    That's just enough time for it
    to settle in to good (close to optimal) behavior.

    """
    )

    bench.run(
        QLearningCuriosity,
        ContextualBandit,
        agent_args={
            "curiosity_scale": 100.0,
            "discount_factor": 0.0,
            "learning_rate": 0.01,
        },
        world_args={"n_time_steps": 20000},
        record=False,
    )

    print(
        """

    A random agent will score an average reward of about 65.
    A perfect score is closer to 110.

    Dig in to README.md for usage, examples, and API documentation.

    """
    )


if __name__ == "__main__":
    run_demo()
