import numpy as np
from myrtle.worlds.base_world import BaseWorld


class OneHotContextualBandit(BaseWorld):
    """
    A multi-armed bandit, except that at each time step the order of the
    bandits is shuffled. The shuffled order is sensed.

    Just like ContextualBandit, except that sensed order is reported as a set of
    one-hot arrays.

    This world tests an agent's ability to use sensor information to determine
    which action to take.
    """

    def __init__(
        self,
        n_loop_steps=1000,
        n_episodes=1,
        loop_steps_per_second=50,
    ):
        self.init_common(
            n_loop_steps=n_loop_steps,
            n_episodes=n_episodes,
            loop_steps_per_second=loop_steps_per_second,
        )
        self.name = "One-hot contextual bandit"

        self.n_sensors = 16
        self.n_actions = 4
        self.n_rewards = 4

        # The highest paying bandit is 2 with average payout of .4 * 280 = 112.
        # Others are 50 or less.
        self.bandit_payouts = [150, 200, 280, 320]
        self.bandit_hit_rates = [0.3, 0.25, 0.4, 0.15]

    def reset(self):
        self.bandit_order = np.arange(self.n_actions)
        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

    def sense(self):
        # Shuffle and sense the order of the bandits.
        self.bandit_order = np.arange(self.n_actions)
        np.random.shuffle(self.bandit_order)
        self.sensors = np.zeros(self.n_sensors)
        for i_position, i_bandit in enumerate(self.bandit_order):
            # Populate the one-hot sensed order
            self.sensors[i_position * self.n_actions + i_bandit] = 1

    def step_world(self):
        # Populate the rewards list.
        self.rewards = [0] * self.n_actions
        for i_position, i_bandit in enumerate(self.bandit_order):
            # For the selected bandits, check whether they pay out
            if np.random.sample() < self.bandit_hit_rates[i_bandit]:
                self.rewards[i_position] = (
                    self.actions[i_position] * self.bandit_payouts[i_bandit]
                )
