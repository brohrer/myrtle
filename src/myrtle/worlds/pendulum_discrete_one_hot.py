import numpy as np
from myrtle.worlds.pendulum import Pendulum


class PendulumDiscreteOneHot(Pendulum):
    """
    Similar to PendulumDiscrete, except that the state is one-hot,
    meaning that it is an array of zeros with exactly one element that is a one.
    Each state element represents a unique combination of a position and
    velocity. The total number of states is the number of possible
    position bins multiplied by the number of possible velocity bins.
    """

    def reset_sensors(self):
        self.name = "Discrete Valued, One-Hot Pendulum"
        self.n_positions = 36

        self.velocity_bins = np.linspace(-15.0, 15.0, 61)
        self.n_velocities = self.velocity_bins.size + 1

        self.n_sensors = self.n_positions * self.n_velocities
        self.sensors = np.zeros(self.n_sensors)

    def step_sensors(self):
        self.sensors = np.zeros(self.n_sensors)

        i_position = int(self.n_positions * self.position / (2 * np.pi))

        try:
            i_velocity = 1 + np.where(self.velocity > self.velocity_bins)[0][-1]
        except IndexError:
            i_velocity = 0

        # The state index is a combination of the position index
        # and the velocity index.
        i_state = i_position * self.n_velocities + i_velocity
        self.sensors[i_state] = 1
