# Longer-running tests, providing a deeper functionality check
from myrtle import bench
from myrtle.agents.base_agent import BaseAgent
# from myrtle.agents.random_single_action import RandomSingleAction
# from myrtle.agents.random_multi_action import RandomMultiAction
# from myrtle.agents.greedy_state_blind import GreedyStateBlind
# from myrtle.agents.greedy_state_blind_eps import GreedyStateBlindEpsilon
# from myrtle.agents.q_learning_eps import QLearningEpsilon
# from myrtle.agents.q_learning_curiosity import QLearningCuriosity

from myrtle.worlds.base_world import BaseWorld
# from myrtle.worlds.stationary_bandit import StationaryBandit
# from myrtle.worlds.nonstationary_bandit import NonStationaryBandit
# from myrtle.worlds.intermittent_reward_bandit import IntermittentRewardBandit
# from myrtle.worlds.contextual_bandit import ContextualBandit
# from myrtle.worlds.one_hot_contextual_bandit import OneHotContextualBandit
# from myrtle.worlds.pendulum import Pendulum
# from myrtle.worlds.pendulum_discrete import PendulumDiscrete


def main():
    # Specify which scenarios to run
    base_world_base_agent()


def base_world_base_agent():
    bench.run(
        BaseAgent,
        BaseWorld,
        do_logging=False,
        world_args={
            "n_loop_steps": 100,
            "n_episodes": 1,
            "loop_steps_per_second": 10,
        },
    )

# bench.run(BaseAgent, StationaryBandit)
# bench.run(RandomSingleAction, StationaryBandit)
# bench.run(RandomSingleAction, ContextualBandit)
# bench.run(RandomMultiAction, StationaryBandit)
# bench.run(GreedyStateBlind, StationaryBandit)
# bench.run(GreedyStateBlindEpsilon, StationaryBandit)
# bench.run(GreedyStateBlindEpsilon, BaseWorld)

# bench.run(GreedyStateBlindEpsilon, NonStationaryBandit)
# bench.run(GreedyStateBlindEpsilon, IntermittentRewardBandit)
# bench.run(GreedyStateBlindEpsilon, ContextualBandit)
# bench.run(
#     GreedyStateBlindEpsilon,
#     OneHotContextualBandit,
#     agent_args={"epsilon": 0.9},
#     world_args={"n_time_steps": 100, "n_episodes": 3},
# )

# bench.run(
#     QLearningEpsilon,
#     ContextualBandit,
#     # agent_args={"epsilon": 0.2, "learning_rate": 0.1, "discount_factor": 0.5},
#     agent_args={"epsilon": 0.2, "learning_rate": 0.001, "discount_factor": 0.0},
#     world_args={"n_time_steps": 100000, "n_episodes": 1},
# )

# bench.run(
#     QLearningEpsilon,
#     OneHotContextualBandit,
#     agent_args={"epsilon": 0.1, "learning_rate": 0.001, "discount_factor": 0.0},
#     world_args={"n_time_steps": 10000, "n_episodes": 1},
# )

# bench.run(
#     RandomSingleAction,
#     OneHotContextualBandit,
#     world_args={"n_time_steps": 1000, "n_episodes": 1},
# )

"""
bench.run(
    QLearningEpsilon,
    ContextualBandit,
    agent_args={"epsilon": 0.1, "learning_rate": 0.01, "discount_factor": 0.0},
    world_args={"n_time_steps": 1000000, "n_episodes": 1},
)
"""
"""
bench.run(
    QLearningCuriosity,
    ContextualBandit,
    agent_args={
        "curiosity_scale": 100.0,
        "discount_factor": 0.0,
        "learning_rate": 0.01,
    },
    world_args={"n_time_steps": 20000},
)
"""

"""
bench.run(
    QLearningCuriosity,
    PendulumDiscrete,
    agent_args={
        "curiosity_scale": 0.1,
        "discount_factor": 0.9,
        "learning_rate": 0.1,
    },
    world_args={"n_time_steps": int(1e10)},
)
"""


if __name__ == "__main__":
    main()
