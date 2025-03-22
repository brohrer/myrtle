import numpy as np
from myrtle.worlds.pendulum import Pendulum


class PendulumDiscrete(Pendulum):
    name = "Discrete Valued Pendulum"

    def reset_sensors(self):
        self.n_positions = 36
        positions = np.zeros(self.n_positions)

        self.velocity_bins = np.linspace(-15.0, 15.0, 61)
        self.n_velocities = self.velocity_bins.size + 1
        velocities = np.zeros(self.n_velocities)

        self.n_sensors = self.n_positions + self.n_velocities
        self.sensors = np.concatenate((positions, velocities))

    def step_sensors(self):
        epsilon = 1e-4
        positions = np.zeros(self.n_positions)
        # Find the discrete position bin.
        # Add a small epsilon to account for weird numerical case where
        # self.position is elmost exactly 2 pi.
        i_position = int(self.n_positions * self.position / (2 * np.pi + epsilon))
        if i_position == self.n_positions:
            i_position = 0
            print(f"position is too big {self.position}")
        positions[i_position] = 1

        velocities = np.zeros(self.n_velocities)
        try:
            i_velocity = 1 + np.where(self.velocity > self.velocity_bins)[0][-1]
        except IndexError:
            i_velocity = 0
        velocities[i_velocity] = 1

        self.sensors = np.concatenate((positions, velocities))
        self.write_pendulum_state()
