import numpy as np
from myrtle.worlds.base_world import BaseWorld


class StationaryBandit(BaseWorld):
    def __init__(
        self,
        n_loop_steps=1000,
        n_episodes=1,
        loop_steps_per_second=100,
    ):
        self.init_common(
            n_loop_steps=n_loop_steps,
            n_episodes=n_episodes,
            loop_steps_per_second=loop_steps_per_second,
        )
        self.name = "Stationary bandit"

        self.n_sensors = 0
        self.n_actions = 5
        self.n_rewards = 5

        # The highest paying bandit is 2 with average payout of .4 * 280 = 112.
        # Others are 100 or less.
        self.bandit_payouts = [150, 200, 280, 320, 500]
        self.bandit_hit_rates = [0.6, 0.5, 0.4, 0.3, 0.2]

    def sense(self):
        self.rewards = [0] * self.n_actions
        for i in range(self.n_actions):
            if np.random.sample() < self.bandit_hit_rates[i]:
                self.rewards[i] = self.actions[i] * self.bandit_payouts[i]
