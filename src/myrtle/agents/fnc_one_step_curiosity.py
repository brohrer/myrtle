import numpy as np
from myrtle.agents.base_agent import BaseAgent
from cartographer.model import NaiveCartographer as Model


class FNCOneStepCuriosity(BaseAgent):
    """
    An agent that uses the Fuzzy Naive Cartographer (FNC) as a world model.

    It also has a basic greedy one-step-lookahead planner and incorporates
    curiosity-driven exploration.

    See https://brandonrohrer.com/cartographer
    """

    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        action_threshold=0.5,
        curiosity_scale=1.0,
        exploitation_factor=1.0,
        feature_decay_rate=0.35,
        reward_update_rate=0.3,
        sensor_q=None,
        action_q=None,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        self.name = "Naive Cartographer with One-Step Lookahead and Curiosity"
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.sensor_q = sensor_q
        self.action_q = action_q

        self.model = Model(
            n_sensors=self.n_sensors,
            n_actions=self.n_actions,
            n_rewards=self.n_rewards,
            feature_decay_rate=feature_decay_rate,
            reward_update_rate=reward_update_rate,
        )

        # A weight that affects how much influence curiosity has on the
        # agent's decision making process. It gets accumulated across all actions,
        # so it gets pre-divided by the number of actions to keep it from being
        # artificially inflated.
        self.curiosity_scale = curiosity_scale / self.n_actions

        # A parameter that affects how quickly the agent settles in to
        # greedily choosing the best known action.
        #   0.0: Always keep exploring (as in epsilon-greedy exploration)
        #   1.0: Explore more at first, then taper off, but stay a bit curious
        #   2.0: After some initial exploration, settle in to exploitation
        # Empirical investigation with a pendulum world suggests that
        # 2.0 gives faster convergence and better overall results
        # in a deterministic world.
        # In a stochastic world, such as one-hot contextual bandit,
        # 1.0 lets the agent experiment for long enough to learn the patterns
        self.exploitation_factor = exploitation_factor

        self.reward_scale = 1.0
        self.reward_scale_update_rate = 0.01

        self.action_threshold = action_threshold

        self.initialize_log(log_name, log_dir, logging_level)

        # How often to report progress
        self.report_steps = int(1e4)

        # This will get incremented to 0 by the reset.
        self.i_episode = -1
        self.reset()

    def reset(self):
        self.display()
        self.sensors = np.zeros(self.n_sensors)
        self.previous_sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards
        self.reward_history = [0.0] * self.report_steps
        self.curiosities = np.zeros((self.n_sensors, self.n_actions + 2))

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

        self.model.update_sensors_and_rewards(self.sensors, self.rewards)

        # Plan using one-step lookahead.
        # Choose a single action to take on this time step by looking ahead
        # to the expected immediate reward it would return, and including
        # any curiosity that would be satisfied.
        predictions, predicted_rewards, uncertainties = self.model.predict()

        # Calculate the curiosity associated with each action.
        # There's a small amount of intrinsic reward associated with
        # satisfying curiosity.
        curiosities = np.max(self.curiosities * self.sensors[:, np.newaxis], axis=0)

        # Find the most valuable action, including the influence of curiosity.
        # Ignore the "average" action from the model.
        # It will always be in the final position.
        i_action = np.argmax((predicted_rewards + curiosities)[:-1])

        # Alternative method--randomly select among actions if there
        # might be more than one best choice.
        # max_value = np.max((predicted_rewards + curiosities)[:-1])
        # Make the most of existing experience.
        # In the case where there are multiple matches for the highest value,
        # randomly pick one of them. This is especially useful
        # in the beginning when all the values are zero.
        # i_action = np.random.choice(
        #     np.where((predicted_rewards + curiosities)[:-1] == max_value)[0]
        # )

        self.actions = np.zeros(self.n_actions)
        # If the "do nothing" has the highest expected value, then do nothing.
        if i_action < self.n_actions:
            self.actions[i_action] = 1

        self.model.update_actions(self.actions)

        # Update the running estimate of the average reward.
        alpha = self.reward_scale_update_rate
        self.reward_scale = self.reward_scale * (1 - alpha) + reward * alpha

        # Update the curiosities--increment them by the uncertainty,
        # raised to the power of the exploitation factor,
        # scaled to match the average reward.
        self.curiosities += (
            uncertainties**self.exploitation_factor
            * self.curiosity_scale
            * self.reward_scale
        )
        # Reset the curiosity counter on the selected state-action pairs.
        self.curiosities[:, i_action] *= 1.0 - self.sensors

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
