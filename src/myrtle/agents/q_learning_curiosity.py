import numpy as np
from myrtle.agents.base_agent import BaseAgent


class QLearningCuriosity(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        action_threshold=0.5,
        curiosity_scale=1.0,
        discount_factor=0.5,
        learning_rate=0.01,
        sensor_q=None,
        action_q=None,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        self.name = "Q-Learning with Curiosity"
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.sensor_q = sensor_q
        self.action_q = action_q

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

        self.initialize_log(log_name, log_dir, logging_level)

        # How often to report progress
        self.report_steps = int(1e4)

        # This will get incremented to 0 by the reset.
        self.i_episode = -1
        self.reset()

        # Store the value table as a dictionary.
        # Keys are sets of sensor readings.
        # Because we can't hash on Numpy arrays for the dict,
        # always use sensor_array.tobytes() as the key.
        self.q_values = {self.previous_sensors.tobytes(): np.zeros(self.n_actions)}

        # Store state-action counts as a dict, too.
        self.counts = {self.previous_sensors.tobytes(): np.zeros(self.n_actions)}
        # And the curiosity associated with each state-action pair as well.
        self.curiosities = {self.previous_sensors.tobytes(): np.zeros(self.n_actions)}

    def reset(self):
        self.display()
        self.sensors = np.zeros(self.n_sensors)
        self.previous_sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards
        self.reward_history = [0.0] * self.report_steps

        self.i_episode += 1
        self.i_step = 0

    def step(self):
        # Update the running total of actions taken and how much reward they generate.
        reward = 0.0
        for reward_channel in self.rewards:
            if reward_channel is not None:
                reward += reward_channel

        self.reward_history.append(reward)
        self.reward_history.pop(0)

        if self.sensors.tobytes() not in self.q_values:
            self.q_values[self.sensors.tobytes()] = np.zeros(self.n_actions)
            self.counts[self.sensors.tobytes()] = np.zeros(self.n_actions)
            self.curiosities[self.sensors.tobytes()] = np.zeros(self.n_actions)

        # Find the maximum expected value to come out of the next action.
        values = self.q_values[self.sensors.tobytes()]
        max_value = np.max(values)

        # Find the actions that were taken.
        # (In it's current implementation, there will never be more than one.)
        try:
            previous_action = np.where(self.actions > self.action_threshold)[0][0]
            if self.counts[self.previous_sensors.tobytes()][previous_action] == 1:
                self.q_values[self.previous_sensors.tobytes()][
                    previous_action
                ] = reward
            else:
                self.q_values[self.previous_sensors.tobytes()][previous_action] = (
                    1 - self.learning_rate
                ) * self.q_values[self.previous_sensors.tobytes()][
                    previous_action
                ] + self.learning_rate * (
                    reward + self.discount_factor * max_value
                )
        except IndexError:
            # Catch the case where there has been no action.
            # This is true for the first iteration.
            pass

        # Calculate the curiosity associated with each action.
        # There's a small amount of intrinsic reward associated with
        # satisfying curiosity.
        count = self.counts[self.sensors.tobytes()]
        uncertainty = 1 / (count**2 + 1)
        # uncertainty = 1 / (count + 1)
        self.curiosities[self.sensors.tobytes()] = (
            self.curiosities[self.sensors.tobytes()]
            + uncertainty * self.curiosity_scale
        )
        curiosity = self.curiosities[self.sensors.tobytes()]

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
        self.curiosities[self.sensors.tobytes()][i_action] = 0
        self.counts[self.sensors.tobytes()][i_action] += 1

        if self.i_step % self.report_steps == 0:
            # self.display()
            pass

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
