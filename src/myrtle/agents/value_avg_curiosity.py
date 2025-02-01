import numpy as np
from myrtle.agents.base_agent import BaseAgent


class ValueAvgCuriosity(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        action_threshold=0.5,
        curiosity_scale=1.0,
    ):
        self.name = "Q-Averages with Curiosity"
        self.init_common(
            n_sensors=n_sensors,
            n_actions=n_actions,
            n_rewards=n_rewards,
        )

        # A weight that affects how much influence curiosity has on the
        # agent's decision making process. It gets accumulated across all actions,
        # so it gets pre-divided by the number of actions to keep it from being
        # artificially inflated.
        self.curiosity_scale = curiosity_scale / self.n_actions

        # Q-Learning assumes that actions are binary \in {0, 1},
        # but just in case a world slips in fractional actions add a threshold.
        self.action_threshold = action_threshold

        # How often to report progress
        # self.report_steps = 1000

        # Store the value table as a dictionary.
        # Keys are sets of sensor readings.
        # Because we can't hash on Numpy arrays for the dict,
        # always use sensor_array.tobytes() as the key.
        self.q_values = {np.zeros(self.n_sensors).tobytes(): np.zeros(self.n_actions)}

        # Store state-action counts as a dict, too.
        self.counts = {np.zeros(self.n_sensors).tobytes(): np.zeros(self.n_actions)}

        # And the curiosity associated with each state-action pair as well.
        self.curiosities = {
            np.zeros(self.n_sensors).tobytes(): np.zeros(self.n_actions)
        }

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.previous_sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards
        # self.reward_history = [0] * self.report_steps

    def choose_action(self):
        # Update the running total of actions taken and how much reward they generate.
        reward = 0.0
        for reward_channel in self.rewards:
            if reward_channel is not None:
                reward += reward_channel

        # self.reward_history.append(reward)
        # self.reward_history.pop(0)

        if self.sensors.tobytes() not in self.q_values:
            self.q_values[self.sensors.tobytes()] = np.zeros(self.n_actions)
            self.counts[self.sensors.tobytes()] = np.zeros(self.n_actions)
            self.curiosities[self.sensors.tobytes()] = np.zeros(self.n_actions)

        # Find the maximum expected value to come out of the next action.
        values = self.q_values[self.sensors.tobytes()]
        max_value = np.max(values)

        # Find the action that was taken. Assume it was only one action.
        # (In it's current implementation, there will never be more than one.)
        try:
            previous_action = np.where(self.actions > self.action_threshold)[0][0]
            previous_count = self.counts[self.previous_sensors.tobytes()][
                previous_action
            ]
            self.q_values[self.previous_sensors.tobytes()][previous_action] = (
                1 - 1 / (previous_count + 1)
            ) * self.q_values[self.previous_sensors.tobytes()][previous_action] + (
                1 / (previous_count + 1) * reward
            )
        except IndexError:
            pass

        # Calculate the curiosity associated with each action.
        # There's a small amount of intrinsic reward associated with
        # satisfying curiosity.
        count = self.counts[self.sensors.tobytes()]
        # uncertainty = 1 / (count + 1)
        uncertainty = 1 / (count**2 + 1)
        self.curiosities[self.sensors.tobytes()] = (
            self.curiosities[self.sensors.tobytes()]
            + uncertainty * self.curiosity_scale
        )
        curiosity = self.curiosities[self.sensors.tobytes()]

        # Find the most valuable action, including the influence of curiosity
        max_value = np.max(values + curiosity)
        # Make the  of existing experience.
        # In the case where there are multiple matches for the highest value,
        # randomly pick one of them. This is especially useful
        # in the beginning when all the values are zero.
        i_action = np.random.choice(np.where((values + curiosity) == max_value)[0])

        self.actions = np.zeros(self.n_actions)
        self.actions[i_action] = 1

        # Reset the curiosity counter on the selected state-action pair.
        self.curiosities[self.sensors.tobytes()][i_action] = 0
        self.counts[self.sensors.tobytes()][i_action] += 1

        # Make sure to make a copy here, so that previous_sensors and sensors don't
        # end up pointing at the same Numpy Array object.
        self.previous_sensors = self.sensors.copy()
