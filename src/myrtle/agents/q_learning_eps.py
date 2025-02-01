import numpy as np
from myrtle.agents.base_agent import BaseAgent


class QLearningEpsilon(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        action_threshold=0.5,
        epsilon=0.2,
        discount_factor=0.5,
        learning_rate=0.01,
    ):
        self.name = "Epsilon-Greedy Q-Learning"
        self.init_common(
            n_sensors=n_sensors,
            n_actions=n_actions,
            n_rewards=n_rewards,
        )

        # A parameter that affects how often the agent chooses to explore
        # random actions, rather than exploit (choose the best known action).
        self.epsilon = epsilon

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # Q-Learning assumes that actions are binary \in {0, 1},
        # but just in case a world slips in fractional actions add a threshold.
        self.action_threshold = action_threshold

        # How often to report progress
        self.report_steps = 1000

        # Store the value table as a dictionary.
        # Keys are sets of sensor readings.
        # Because we can't hash on Numpy arrays for the dict,
        # always use sensor_array.tobytes() as the key.
        self.q_values = {np.zeros(self.n_sensors).tobytes(): np.zeros(self.n_actions)}

    def reset(self):
        self.display()
        self.sensors = np.zeros(self.n_sensors)
        self.previous_sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards
        self.reward_history = [0] * self.report_steps

    def choose_action(self):
        # Update the running total of actions taken and how much reward they generate.
        reward = 0.0
        for reward_channel in self.rewards:
            if reward_channel is not None:
                reward += reward_channel

        self.reward_history.append(reward)
        self.reward_history.pop(0)

        if self.sensors.tobytes() not in self.q_values:
            self.q_values[self.sensors.tobytes()] = np.zeros(self.n_actions)

        # Find the maximum expected value to come out of the next action.
        values = self.q_values[self.sensors.tobytes()]
        max_value = np.max(values)

        # Find the actions that were taken.
        # (In it's current implementation, there will never be more than one.)
        try:
            previous_action = np.where(self.actions > self.action_threshold)[0][0]
            self.q_values[self.previous_sensors.tobytes()][previous_action] = (
                1 - self.learning_rate
            ) * self.q_values[self.previous_sensors.tobytes()][
                previous_action
            ] + self.learning_rate * (reward + self.discount_factor * max_value)
        except IndexError:
            # Catch the case where there has been no action.
            # This is true for the first iteration.
            pass

        if np.random.sample() > self.epsilon:
            # Recalculate in case `values` got modified during the update.
            max_value = np.max(values)
            # Make the most of existing experience.
            # In the case where there are multiple matches for the highest value,
            # randomly pick one of them. This is especially useful
            # in the beginning when all the values are zero.
            i_action = np.random.choice(np.where(values == max_value)[0])
            # print(i_action, "||", values)
        else:
            # Explore to gain new experience
            i_action = np.random.choice(self.n_actions)

        self.actions = np.zeros(self.n_actions)
        self.actions[i_action] = 1

        if self.i_step % self.report_steps == 0:
            self.display()

        # Make sure to make a copy here, so that previous_sensors and sensors don't
        # end up pointing at the same Numpy Array object.
        self.previous_sensors = self.sensors.copy()

    def display(self):
        try:
            if self.i_step == 0:
                return
            n = np.minimum(self.i_step, self.report_steps)
            avg_reward = np.sum(np.array(self.reward_history)) / n
        except AttributeError:
            return

        print(
            f"Average reward of {avg_reward} at time step {self.i_step:,},"
            + f" episode {self.i_episode}"
        )
        n_lines = 4
        for _ in range(n_lines):
            print()
