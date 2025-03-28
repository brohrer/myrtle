"""
Longer-running tests, providing a deeper functionality check
"""

import os
import time
from sqlogging import logging
from myrtle import bench
from myrtle.config import log_directory

from myrtle.agents.base_agent import BaseAgent
from myrtle.agents.random_multi_action import RandomMultiAction
from myrtle.agents.greedy_state_blind import GreedyStateBlind
from myrtle.agents.greedy_state_blind_eps import GreedyStateBlindEpsilon
from myrtle.agents.value_avg_curiosity import ValueAvgCuriosity
from myrtle.agents.q_learning_eps import QLearningEpsilon
from myrtle.agents.q_learning_curiosity import QLearningCuriosity

from myrtle.worlds.base_world import BaseWorld
from myrtle.worlds.stationary_bandit import StationaryBandit
from myrtle.worlds.nonstationary_bandit import NonStationaryBandit
from myrtle.worlds.intermittent_reward_bandit import IntermittentRewardBandit
from myrtle.worlds.contextual_bandit import ContextualBandit
from myrtle.worlds.one_hot_contextual_bandit import OneHotContextualBandit
from myrtle.worlds.pendulum_discrete import PendulumDiscrete
from myrtle.worlds.pendulum import Pendulum

_test_db_name = f"temp_integration_test_{int(time.time())}"
_default_timeout = 30 * 60  # in seconds
_long_timeout = 120 * 60  # in seconds


def main():
    # Specify which scenarios to run
    # test_base_world_base_agent()
    # test_base_world_random_multi_action_agent()
    # test_base_world_greedy_state_blind_agent()
    # test_base_world_greedy_state_blind_eps_agent()
    # test_base_world_value_avg_curiosity_agent()
    # test_base_world_q_learning_eps_agent()
    test_base_world_q_learning_curiosity_agent()
    # test_stationary_bandit_world_q_learning_curiosity_agent()
    # test_nonstationary_bandit_world_q_learning_curiosity_agent()
    # test_intermittent_reward_bandit_world_q_learning_curiosity_agent()
    # test_contextual_bandit_world_q_learning_curiosity_agent()
    # test_one_hot_contextual_bandit_world_q_learning_curiosity_agent()
    # test_pendulum_discrete_world_q_learning_curiosity_agent()
    # test_pendulum_world_q_learning_curiosity_agent()


def db_cleanup():
    db_filename = f"{_test_db_name}.db"
    db_path = os.path.join(log_directory, db_filename)
    os.remove(db_path)


def run_world_with_agent(
    world_class,
    agent_class,
    n_loop_steps=1000,
    n_episodes=3,
    loops_per_second=20,
    # loops_per_second=40,
    agent_args={},
    reward_lower_bound=-1.0,
    reward_upper_bound=0.0,
    timeout=_default_timeout,
):
    """
    For a given agent class, run it against a BaseWorld
    """
    start_time = time.time()
    exitcode = bench.run(
        agent_class,
        world_class,
        log_to_db=True,
        logging_db_name=_test_db_name,
        timeout=timeout,
        world_args={
            "n_loop_steps": n_loop_steps,
            "n_episodes": n_episodes,
            "loop_steps_per_second": loops_per_second,
            "verbose": True,
        },
        agent_args=agent_args,
    )
    assert exitcode == 0

    run_time = time.time() - start_time
    timed_out = run_time > timeout * 0.99
    print()
    print(f"Ran in {int(run_time)} seconds")

    assert not timed_out

    if not timed_out:
        logger = logging.open_logger(
            name=_test_db_name,
            dir_name=log_directory,
            level="info",
        )
        result = logger.query(
            f"""
            SELECT AVG(reward)
            FROM {_test_db_name}
            GROUP BY episode
            ORDER BY episode DESC
        """
        )
        print(f"Average reward: {result[1][0]}")

        assert result[1][0] > reward_lower_bound
        assert result[1][0] < reward_upper_bound
    else:
        print("Run timed out before it could complete")

    print()
    db_cleanup()


def test_base_world_base_agent():
    run_world_with_agent(BaseWorld, BaseAgent)


def test_base_world_random_multi_action_agent():
    run_world_with_agent(BaseWorld, RandomMultiAction)


def test_base_world_greedy_state_blind_agent():
    run_world_with_agent(
        BaseWorld,
        GreedyStateBlind,
        reward_lower_bound=-0.3,
        reward_upper_bound=0.3,
    )


def test_base_world_greedy_state_blind_eps_agent():
    run_world_with_agent(
        BaseWorld,
        GreedyStateBlindEpsilon,
        reward_lower_bound=-0.3,
        reward_upper_bound=0.3,
    )


def test_base_world_value_avg_curiosity_agent():
    run_world_with_agent(
        BaseWorld,
        ValueAvgCuriosity,
        reward_lower_bound=-0.3,
        reward_upper_bound=0.3,
    )


def test_base_world_q_learning_eps_agent():
    run_world_with_agent(
        BaseWorld,
        QLearningEpsilon,
        reward_lower_bound=-0.3,
        reward_upper_bound=0.3,
    )


def test_base_world_q_learning_curiosity_agent():
    run_world_with_agent(
        BaseWorld,
        QLearningCuriosity,
        reward_lower_bound=-0.3,
        reward_upper_bound=0.3,
    )


def test_stationary_bandit_world_q_learning_curiosity_agent():
    run_world_with_agent(
        StationaryBandit,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        reward_lower_bound=10.0,
        reward_upper_bound=100.0,
        timeout=_long_timeout,
    )


def test_nonstationary_bandit_world_q_learning_curiosity_agent():
    run_world_with_agent(
        NonStationaryBandit,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        reward_lower_bound=50.0,
        reward_upper_bound=150.0,
        timeout=_long_timeout,
    )


def test_intermittent_reward_bandit_world_q_learning_curiosity_agent():
    run_world_with_agent(
        IntermittentRewardBandit,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        reward_lower_bound=50.0,
        reward_upper_bound=150.0,
        timeout=_long_timeout,
    )


def test_contextual_bandit_world_q_learning_curiosity_agent():
    # agent_args={"epsilon": 0.2, "learning_rate": 0.001, "discount_factor": 0.0},
    # world_args={"n_time_steps": 100000, "n_episodes": 1},
    run_world_with_agent(
        ContextualBandit,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        reward_lower_bound=5.0,
        reward_upper_bound=150.0,
        timeout=_long_timeout,
    )


def test_one_hot_contextual_bandit_world_q_learning_curiosity_agent():
    # agent_args={"epsilon": 0.2, "learning_rate": 0.001, "discount_factor": 0.0},
    # world_args={"n_time_steps": 100000, "n_episodes": 1},
    run_world_with_agent(
        OneHotContextualBandit,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        reward_lower_bound=5.0,
        reward_upper_bound=100.0,
        timeout=_long_timeout,
    )


def test_pendulum_world_q_learning_curiosity_agent():
    agent_args = {
        "curiosity_scale": 0.1,
        "discount_factor": 0.9,
        "learning_rate": 0.1,
    }
    run_world_with_agent(
        Pendulum,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        agent_args=agent_args,
        reward_lower_bound=0.0,
        reward_upper_bound=1.0,
        timeout=_long_timeout,
    )


def test_pendulum_discrete_world_q_learning_curiosity_agent():
    agent_args = {
        "curiosity_scale": 0.1,
        "discount_factor": 0.9,
        "learning_rate": 0.1,
    }
    run_world_with_agent(
        PendulumDiscrete,
        QLearningCuriosity,
        n_loop_steps=int(1e4),
        agent_args=agent_args,
        reward_lower_bound=0.0,
        reward_upper_bound=1.0,
        timeout=_long_timeout,
    )


if __name__ == "__main__":
    main()
