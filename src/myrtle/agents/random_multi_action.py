import numpy as np
from myrtle.agents.base_agent import BaseAgent


class RandomMultiAction(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        avg_actions=2.0,
        sensor_q=None,
        action_q=None,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        self.name = "Random Multi-Action"
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.sensor_q = sensor_q
        self.action_q = action_q

        self.action_prob = avg_actions / self.n_actions

        self.initialize_log(log_name, log_dir, logging_level)

        # This will get incremented to 0 by the reset.
        self.i_episode = -1
        self.reset()

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.rewards = [0] * self.n_rewards
        self.actions = np.zeros(self.n_actions)
        self.i_episode += 1
        self.i_step = 0

    def step(self):
        # Pick whether to include each action independently
        self.actions = np.random.choice(
            [0, 1],
            size=self.n_actions,
            p=[1 - self.action_prob, self.action_prob],
        )
