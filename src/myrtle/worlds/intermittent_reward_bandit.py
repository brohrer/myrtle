import numpy as np
from myrtle.worlds.base_world import BaseWorld
from pacemaker.pacemaker import Pacemaker


class IntermittentRewardBandit(BaseWorld):
    def __init__(
        self,
        n_time_steps=1000,
        n_episodes=1,
        sensor_q=None,
        action_q=None,
        report_q=None,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        # Initialize constants
        self.n_sensors = 0
        self.n_actions = 5
        self.n_rewards = 5
        self.steps_per_second = 100
        # Number of time steps to run in a single episode
        self.n_time_steps = n_time_steps
        self.n_episodes = n_episodes
        self.name = "Intermittent bandit"

        # This gets incremented to 0 with the first reset(), before the run starts.
        self.i_episode = -1

        self.sensor_q = sensor_q
        self.action_q = action_q
        self.report_q = report_q

        self.pm = Pacemaker(self.steps_per_second)
        self.initialize_log(log_name, log_dir, logging_level)

        # The highest paying bandit is 2 with average payout of .4 * 280 = 112.
        # Others are 100 or less.
        self.bandit_payouts = [150, 200, 280, 320, 500]
        self.bandit_hit_rates = [0.6, 0.5, 0.4, 0.3, 0.2]

        # The fraction of the time, on average, that a given reward signal
        # will be None
        self.intermittency = 0.1

    def reset(self):
        # This block will probably be needed in the reset() of every world.
        ####
        self.i_step = 0
        if self.i_episode > 0:
            self.sensor_q.put({"truncated": True})
        ####

        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

    def step(self):
        print(
            f"step {self.i_step}, episode {self.i_episode}              ",
            end="\r",
        )
        self.rewards = [0] * self.n_actions
        for i in range(self.n_actions):
            if np.random.sample() < self.bandit_hit_rates[i]:
                self.rewards[i] = self.actions[i] * self.bandit_payouts[i]

        # Intermittently blank out reward signals
        for i in range(self.n_rewards):
            if np.random.sample() < self.intermittency:
                self.rewards[i] = None
