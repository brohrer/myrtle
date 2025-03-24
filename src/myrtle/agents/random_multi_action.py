import numpy as np
from myrtle.agents.base_agent import BaseAgent


class RandomMultiAction(BaseAgent):
    name = "Random Multi-Action"

    def __init__(
        self,
        avg_actions=2.0,
        **kwargs,
    ):
        self.init_common(**kwargs)

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
