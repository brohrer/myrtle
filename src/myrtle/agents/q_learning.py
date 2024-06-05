import numpy as np
from myrtle.agents.base_agent import BaseAgent


class QLearningEps(BaseAgent):
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
        self.name = "Epsilon-Greedy Q-Learning"
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.sensor_q = sensor_q
        self.action_q = action_q

        self.epsilon = 0.1

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

        self.q_values = np.zeros((self.n_sensors, self.n_actions))
        self.learning_rate = 0.1
        self.discount_factor = 0.8

    def step(self):
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
