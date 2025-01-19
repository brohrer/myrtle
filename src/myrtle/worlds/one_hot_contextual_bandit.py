import numpy as np
from myrtle.worlds.base_world import BaseWorld
from pacemaker.pacemaker import Pacemaker


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
        n_time_steps=1000,
        n_episodes=1,
        steps_per_second=50,
    ):
        # Initialize constants
        self.n_sensors = 16
        self.n_actions = 4
        self.n_rewards = 4
        self.steps_per_second = steps_per_second

        # Number of time steps to run in a single episode
        self.n_time_steps = n_time_steps
        self.n_episodes = n_episodes

        self.name = "One-hot contextual bandit"

        # This gets incremented to 0 with the first reset(), before the run starts.
        self.i_episode = -1

        # self.sensor_q = sensor_q
        # self.action_q = action_q
        # self.report_q = report_q

        self.pm = Pacemaker(self.steps_per_second)
        # self.initialize_log(log_name, log_dir, logging_level)

        # Initialize message queue connection.
        self.mq = dsmq.connect_to_server(mq_host, mq_port)

        # The highest paying bandit is 2 with average payout of .4 * 280 = 112.
        # Others are 50 or less.
        self.bandit_payouts = [150, 200, 280, 320]
        self.bandit_hit_rates = [0.3, 0.25, 0.4, 0.15]

    def reset(self):
        # This block will probably be needed in the reset() of every world.
        ####
        self.i_step = 0
        if self.i_episode > 0:
            # self.sensor_q.put({"truncated": True})
            self.mq.put("control", "truncated")
        ####

        self.bandit_order = np.arange(self.n_actions)
        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

    def step(self):
        print(
            f"step {self.i_step}, episode {self.i_episode}              ",
            end="\r",
        )
        self.pm.beat()
        self.read_action_q()

        # Populate the rewards list.
        self.rewards = [0] * self.n_actions
        for i_position, i_bandit in enumerate(self.bandit_order):
            # For the selected bandits, check whether they pay out
            if np.random.sample() < self.bandit_hit_rates[i_bandit]:
                self.rewards[i_position] = (
                    self.actions[i_position] * self.bandit_payouts[i_bandit]
                )

        # Shuffle and sense the order of the bandits in preparation
        # for the next iteration.
        self.bandit_order = np.arange(self.n_actions)
        np.random.shuffle(self.bandit_order)

        self.sensors = np.zeros(self.n_sensors)
        for i_position, i_bandit in enumerate(self.bandit_order):
            # Populate the one-hot sensed order
            self.sensors[i_position * self.n_actions + i_bandit] = 1
