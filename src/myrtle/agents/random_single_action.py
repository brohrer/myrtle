import numpy as np
from myrtle.agents.base_agent import BaseAgent


class RandomSingleAction(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        sensor_q=None,
        action_q=None,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        self.name = "Random Single Action"
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.sensor_q = sensor_q
        self.action_q = action_q

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
        # Pick a random action.
        self.actions = np.zeros(self.n_actions)
        i_action = np.random.choice(self.n_actions)
        self.actions[i_action] = 1
