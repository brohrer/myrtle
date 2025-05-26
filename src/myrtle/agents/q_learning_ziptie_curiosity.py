import json
import numpy as np
from myrtle.agents.base_agent import BaseAgent
from ziptie.algo import Ziptie


class QLearningZiptieCuriosity(BaseAgent):
    name = "Q-Learning and Ziptie with Curiosity"

    def __init__(
        self,
        action_threshold=0.5,
        curiosity_scale=1.0,
        discount_factor=0.5,
        learning_rate=0.01,
        n_features=None,
        ziptie_threshold=100.0,
        **kwargs,
    ):
        self.init_common(**kwargs)
        if n_features is None:
            self.n_max_features = self.n_sensors
        else:
            self.n_max_features = n_features

        self.ziptie = Ziptie(n_cables=self.n_sensors, threshold=ziptie_threshold)

        # A weight that affects how much influence curiosity has on the
        # agent's decision making process. It gets accumulated across all actions,
        # so it gets pre-divided by the number of actions to keep it from being
        # artificially inflated.
        self.curiosity_scale = curiosity_scale / self.n_actions

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # Q-Learning assumes that actions are binary \in {0, 1},
        # but just in case a world slips in fractional actions add a threshold.
        self.action_threshold = action_threshold

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
        self.features = np.zeros(self.n_max_features)
        self.previous_state = self.features.tobytes()
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

    def choose_action(self):
        # Update the running total of actions taken and how much reward they generate.
        reward = 0.0
        for reward_channel in self.rewards:
            if reward_channel is not None:
                reward += reward_channel

        if self.ziptie.n_bundles < self.n_max_features:
            self.ziptie.create_new_bundles()
            self.ziptie.grow_bundles()

        features = self.ziptie.update_bundles(self.sensors)

        ziptie_update_frequency = 100
        if self.i_step % ziptie_update_frequency == 0:
            msg = json.dumps(
                {
                    "step": self.i_step,
                    "episode": self.i_episode,
                    "n_sensors": self.n_sensors,
                    "n_bundles": self.n_max_features,
                    "agglomeration_threshold": self.ziptie.agglomeration_threshold,
                    "nucleation_threshold": self.ziptie.nucleation_threshold,
                    "mapping": self.ziptie.mapping.tolist(),
                    "nucleation": self.ziptie.nucleation_energy.tolist(),
                    "agglomeration": self.ziptie.agglomeration_energy.tolist(),
                }
            )
            self.mq.put("ziptie", msg)

        self.features = np.zeros(self.n_max_features)
        self.features[: features.size] = features

        state = self.features.tobytes()
        # Avoid treating state as concatenated sensors + features with Q-learning.
        # Q-learning combines sensors to get state already, so that makes the
        # ziptie redundant.
        # The only reason to pair ziptie with Q-learning is to test
        # whether ziptie is working as desired.
        # state = np.concatenate((self.sensors, self.features)).tobytes()

        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.n_actions)
            self.counts[state] = np.zeros(self.n_actions)
            self.curiosities[state] = np.zeros(self.n_actions)

        # Find the maximum expected value to come out of the next action.
        values = self.q_values[state]
        max_value = np.max(values)

        # Find the actions that were taken.
        # (In it's current implementation, there will never be more than one.)
        try:
            previous_action = np.where(self.actions > self.action_threshold)[0][0]
            if self.counts[self.previous_state][previous_action] == 0:
                self.q_values[self.previous_state][previous_action] = (
                    reward + self.discount_factor * max_value
                )
            else:
                self.q_values[self.previous_state][previous_action] = (
                    1 - self.learning_rate
                ) * self.q_values[self.previous_state][
                    previous_action
                ] + self.learning_rate * (reward + self.discount_factor * max_value)
        except IndexError:
            # Catch the case where there has been no action.
            # This is true for the first iteration.
            pass

        # Calculate the curiosity associated with each action.
        # There's a small amount of intrinsic reward associated with
        # satisfying curiosity.
        count = self.counts[state]
        # uncertainty = 1 / (np.minimum(count, 1000) ** .5 + 1)
        # uncertainty = 1 / (count ** .5 + 1)
        # uncertainty = 1 / (count**2 + 1)
        uncertainty = 1 / (count + 1)
        self.curiosities[state] = (
            self.curiosities[state] + uncertainty * self.curiosity_scale
        )
        curiosity = self.curiosities[state]

        # Find the most valuable action, including the influence of curiosity
        max_value = np.max(values + curiosity)
        # Make the most of existing experience.
        # In the case where there are multiple matches for the highest value,
        # randomly pick one of them. This is especially useful
        # in the beginning when all the values are zero.
        i_action = np.random.choice(np.where((values + curiosity) == max_value)[0])

        self.actions = np.zeros(self.n_actions)
        self.actions[i_action] = 1

        # Reset the curiosity counter on the selected state-action pair.
        self.curiosities[state][i_action] = 0
        self.counts[state][i_action] += 1

        # Make sure to make a copy here, so that previous_sensors and sensors don't
        # end up pointing at the same Numpy Array object.
        # self.previous_sensors = self.sensors_and_features.copy()
        self.previous_state = state
