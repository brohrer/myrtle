import numpy as np
from myrtle.worlds.pendulum import Pendulum


class PendulumDiscrete(Pendulum):
    def reset_sensors(self):
        self.name = "Discrete Valued Pendulum"
        self.n_positions = 36
        positions = np.zeros(self.n_positions)

        self.velocity_bins = np.linspace(-15.0, 15.0, 61)
        self.n_velocities = self.velocity_bins.size + 1
        velocities = np.zeros(self.n_velocities)

        self.n_sensors = self.n_positions + self.n_velocities
        self.sensors = np.concatenate((positions, velocities))

    def step_sensors(self):
        positions = np.zeros(self.n_positions)
        i_position = int(self.n_positions * self.position / (2 * np.pi))
        positions[i_position] = 1

        velocities = np.zeros(self.n_velocities)
        try:
            i_velocity = 1 + np.where(self.velocity > self.velocity_bins)[0][-1]
        except IndexError:
            i_velocity = 0
        velocities[i_velocity] = 1

        self.sensors = np.concatenate((positions, velocities))
