import numpy as np
from myrtle.agents.base_agent import BaseAgent


class RandomMultiAction(BaseAgent):
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        avg_actions=2.0,
    ):
        self.name = "Random Multi-Action"
        self.init_common(
            n_sensors=n_sensors,
            n_actions=n_actions,
            n_rewards=n_rewards,
        )

        # Convert the average number of actions taken per step to a
        # probability of each action being selected individually.
        self.action_prob = (
            avg_actions
            /
            # Handle the case where avg_actions >= n_actions
            np.maximum(self.n_actions, avg_actions + 1)
        )

    def choose_action(self):
        # Pick whether to include each action independently
        self.actions = np.random.choice(
            [0, 1],
            size=self.n_actions,
            p=[1 - self.action_prob, self.action_prob],
        )
