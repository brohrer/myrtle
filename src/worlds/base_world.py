# pip install pacemaker-lite 
from pacemaker.pacemaker import Pacemaker

class BaseWorld:
    def __init__(self, sensor_q=None, action_q=None):
        self.sensor_q = sensor_q
        self.action_q = action_q

        n_sensors = 4
        n_actions = 3
        n_rewards = 2

        self.max_steps = 100
        self.steps_per_second = 5
        self.pm = Pacemaker(self.steps_per_second)

    def run(self):

        for i_step in range(self.max_steps):
            overtime = self.pm.beat()

            # Read q
            action_q.get

            sensors, reward = self.step(action)

            # Write to q
            sensor_q.put({"sensors": sensors, "reward": reward}) 

    def step(self, action):

        return sensors, reward
