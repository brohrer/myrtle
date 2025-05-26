import numpy as np
from myrtle.worlds.base_world import BaseWorld


class ContextualBandit2D(BaseWorld):
    """
    A multi-armed bandit, except that at each time step the order of the
    bandits is shuffled. The shuffled order is sensed.

    The order is determined by the interaction of two different binary
    variables:

            x2 0     1
         x1  ____________
          0  |ABCD  DABC
          1  |CDAB  BCDA

    This world tests an agent's ability to use sensor information to determine
    which action to take.
    """

    name = "Contextual bandit"

    def __init__(
        self,
        n_loop_steps=1000,
        n_episodes=1,
        loop_steps_per_second=50,
        **kwargs,
    ):
        self.init_common(
            n_loop_steps=n_loop_steps,
            n_episodes=n_episodes,
            loop_steps_per_second=loop_steps_per_second,
            **kwargs,
        )
        self.n_sensors = 4
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
        x1 = np.random.choice(2)
        x2 = np.random.choice(2)
        self.bandit_order = np.roll(np.arange(self.n_actions), int(2 * x1 + x2))
        self.sensors = np.zeros(4)
        self.sensors[x1] = 1.0
        self.sensors[x2 + 2] = 1.0

    def step_world(self):
        # Calculate the reward based on the shuffled order of the previous time step.
        self.rewards = [0] * self.n_actions
        for i in range(self.n_actions):
            if np.random.sample() < self.bandit_hit_rates[self.bandit_order[i]]:
                self.rewards[i] = (
                    self.actions[i] * self.bandit_payouts[self.bandit_order[i]]
                )
