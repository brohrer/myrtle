import numpy as np
from myrtle.agents.base_agent import BaseAgent


class GreedyStateBlindEpsilon(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        epsilon=0.1,
    ):
        self.name = "Epsilon-Greedy State-Blind"
        self.init_common(
            n_sensors=n_sensors,
            n_actions=n_actions,
            n_rewards=n_rewards,
        )

        self.epsilon = epsilon

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

        if np.random.sample() > self.epsilon:
            # Make the most of existing experience
            return_rate = self.total_return / self.action_count
            i_action = np.argmax(return_rate)
        else:
            # Explore to gain new experience
            i_action = np.random.choice(self.n_actions)

        self.actions = np.zeros(self.n_actions)
        self.actions[i_action] = 1
