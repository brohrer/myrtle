import numpy as np
from myrtle.agents.base_agent import BaseAgent


class QLearningEpsilon(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        action_threshold=0.5,
        epsilon=0.1,
        discount_factor=0.8,
        learning_rate=0.1,
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

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # Q-Learning assumes that actions are binary \in {0, 1},
        # but just in case a world slips in fractional actions add a threshold.
        self.action_threshold = action_threshold

        self.initialize_log(log_name, log_dir, logging_level)

        # This will get incremented to 0 by the reset.
        self.i_episode = -1
        self.reset()

        # Store the value table as a dictionary.
        # Keys are sets of sensor readings.
        # Because we can't hash on Numpy arrays for the dict,
        # always use sensor_array.tobytes() as the key.
        self.q_values = {self.previous_sensors.tobytes(): np.zeros(self.n_actions)}

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.previous_sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

        self.i_episode += 1
        self.i_step = 0

    def step(self):
        # Update the running total of actions taken and how much reward they generate.
        reward = 0.0
        for reward_channel in self.rewards:
            if reward_channel is not None:
                reward += reward_channel

        if self.sensors.tobytes() not in self.q_values:
            self.q_values[self.sensors.tobytes()] = np.zeros(self.n_actions)

        # Find the maximum expected value to come out of the next action.
        values = self.q_values[self.sensors.tobytes()]
        max_value = np.max(values)

        # Find the actions that were taken.
        i_actions = np.where(self.actions > self.action_threshold)[0]
        # Handle actions one at a time, when there is more than one,
        # making the assumption that they operate independently of each other.
        # (In it's current implementation, there will never be more than one.)
        for i_action in i_actions:
            self.q_values[self.previous_sensors.tobytes()][i_action] = (
                1 - self.learning_rate
            ) * self.q_values[self.previous_sensors.tobytes()][
                i_action
            ] + self.learning_rate * (
                reward + self.discount_factor * max_value
            )

        if np.random.sample() > self.epsilon:
            # Recalculate in case `values` got modified during the update.
            max_value = np.max(values)
            # Make the most of existing experience.
            # In the case where there are multiple matches for the highest value,
            # randomly pick one of them. This is especially useful
            # in the beginning when all the values are zero.
            i_action = np.random.choice(np.where(values == max_value)[0])
            print(i_action, "||", values)
        else:
            # Explore to gain new experience
            i_action = np.random.choice(self.n_actions)

        self.actions = np.zeros(self.n_actions)
        self.actions[i_action] = 1

        # Make sure to make a copy here, so that previous_sensors and sensors don't
        # end up pointing at the same Numpy Array object.
        self.previous_sensors = self.sensors.copy()
