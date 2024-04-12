import numpy as np
from pacemaker.pacemaker import Pacemaker

class BaseWorld:
    def __init__(self, sensor_q=None, action_q=None):
        self.sensor_q = sensor_q
        self.action_q = action_q

        self.n_sensors = 4
        self.n_actions = 3
        self.n_rewards = 2

        self.max_steps = 100
        self.steps_per_second = 5
        self.pm = Pacemaker(self.steps_per_second)

        self.reset()

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(sels.n_actions)
        self.rewards = [0] * self.n_rewards

    def run(self):

        for i_step in range(self.max_steps):

            self.step(action)

            # Write to q
            sensor_q.put({"sensors": self.sensors, "reward": self.reward})
            overtime = self.pm.beat()

            # Read q
            # If all goes well, there should be exactly one action
            # array in the queue.
            # If multiple, use the last and ignore others.
            # If none, use an all-zeros action. 
            action_q.get()
            self.actions = np.zeros(sels.n_actions)


    def step(self):
        # use
        self.actions
        # to update
        self.sensors
        self.reward
