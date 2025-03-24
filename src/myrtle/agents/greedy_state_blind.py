"""
Selects the action with the highest historical return.
Doesn't look at the sensors at all when making this decision.
"""

import numpy as np
from myrtle.agents.base_agent import BaseAgent


class GreedyStateBlind(BaseAgent):
    name = "Greedy State-Blind"

    def __init__(self, **kwargs):
        self.init_common(**kwargs)

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.rewards = [0] * self.n_rewards
        self.actions = np.zeros(self.n_actions)

        # Initialize these as ones to avoid any numerical wonkery.
        self.total_return = np.ones(self.n_actions)
        self.action_count = np.ones(self.n_actions)

    def choose_action(self):
        # Update the running total of actions taken and how much reward they generate.
        reward = 0.0
        for reward_channel in self.rewards:
            if reward_channel is not None:
                reward += reward_channel
        reward_by_action = reward * self.actions
        self.total_return += reward_by_action
        self.action_count += self.actions

        self.actions = np.zeros(self.n_actions)
        return_rate = self.total_return / self.action_count
        i_action = np.argmax(return_rate)
        self.actions[i_action] = 1
