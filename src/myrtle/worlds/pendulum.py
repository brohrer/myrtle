import time
import numpy as np
from myrtle.worlds.base_world import BaseWorld
from myrtle.worlds.tools.ring_buffer import RingBuffer
from pacemaker.pacemaker import Pacemaker


class Pendulum(BaseWorld):
    """
    A pendulum that starts at rest. Reward comes from keeping the pendulum
    elevated. Inverting it is optimal.
    """

    def __init__(
        self,
        sensor_q=None,
        action_q=None,
        report_q=None,
        n_time_steps=1000,
        n_episodes=1,
        speedup=8,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        # Positive actions are counter-clockwise torque
        # Negative actions are clockwise torque in Newton-meters
        self.action_scale = 8 * np.array([
            -1.0, -0.75, -0.5, -0.375, -0.25, -0.125,
            0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0,
        ])
        self.n_actions = self.action_scale.size

        self.n_rewards = 1

        # The physics simulation will run at one clock rate, and the interactions
        # with the agent will run at another. For convenience, ensure that
        # the agent interaction steps contain a fixed and constant number of
        # simulation time steps.
        approx_steps_per_second = 4
        self.sim_steps_per_second = 64
        # self.sim_steps_per_second = 128

        self.dt = 1 / float(self.sim_steps_per_second)
        self.sim_steps_per_step = int(
            self.sim_steps_per_second / approx_steps_per_second
        )
        self.steps_per_second = self.sim_steps_per_second / self.sim_steps_per_step

        approx_displays_per_second = 30
        self.sim_steps_per_display = int(
            self.sim_steps_per_second / approx_displays_per_second
        )
        self.reward_smoothing = 0.003

        # Number of time steps to run in a single episode
        self.n_time_steps = n_time_steps
        self.n_episodes = n_episodes

        self.name = "Pendulum"

        self.mass = 1  # kilogram
        self.length = 2  # meter
        self.inertia = self.mass * self.length ** 2 / 12
        self.gravity = -9.8  # meters / second^2
        self.friction = -0.30  # Newton-meters-seconds / radian

        impulse_length = self.sim_steps_per_step
        self.impulse = np.ones(impulse_length)

        # This gets incremented to 0 with the first reset(), before the run starts.
        self.i_episode = -1

        self.sensor_q = sensor_q
        self.action_q = action_q
        self.report_q = report_q

        self.pm = Pacemaker(self.sim_steps_per_second * speedup)
        self.initialize_log(log_name, log_dir, logging_level)

        self.reset()

    def reset(self):
        # This block or something like it will probably be needed in
        # the reset() of every world.
        ####
        self.i_step = 0
        if self.i_episode > 0:
            self.sensor_q.put({"truncated": True})
        ####
        self.i_sim_step = 0

        # Position convention:
        #     0 radians is straight down,
        #     pi / 2 radians is to the right
        #     pi radians is stratight up
        self.position = 0  # radians
        self.velocity = 0  # radians per second

        self.reset_sensors()

        self.torque_buffer = RingBuffer(self.sim_steps_per_step)

        # Action convention: [counter-clockwise torque, clockwise torque, no torque]
        self.actions = np.zeros(self.n_actions)

        self.rewards = [0] * self.n_rewards
        self.smoothed_reward = 0.0

    def reset_sensors(self):
        self.n_sensors = 2

        self.sensors = np.array([self.position, self.velocity])

    def step(self):
        for _ in range(self.sim_steps_per_step):
            # This block or something like it will probably be needed in
            # the step() of every world.
            ####
            self.pm.beat()
            self.i_sim_step += 1
            ####

            # Add commanded actions to the torque buffer.
            # Trying on every sim step make it *almost* instantaneous.
            # There will always be at least a ~1 sim step delay
            # self.read_torque_q()
            if not self.action_q.empty():
                msg = self.action_q.get()
                self.actions = msg["actions"]
                torque_magnitude = np.sum(self.actions * self.action_scale)
                self.torque_buffer.add(torque_magnitude * self.impulse)

            applied_torque = self.torque_buffer.pop()

            # Add in the effect of gravity.
            moment_arm = np.sin(self.position) * self.length / 2
            gravity_torque = self.mass * self.gravity * moment_arm

            # Add in the effect of friction at the bearings.
            friction_torque = self.friction * self.velocity
            torque = applied_torque + gravity_torque + friction_torque

            # Add the discrete-time approximation of Newtonian mechanics, F = ma
            self.velocity += torque * self.dt / self.inertia
            self.position += self.velocity * self.dt

            # Keep position in the range of [0, 2 pi)
            self.position = np.mod(self.position, 2 * np.pi)

            if self.i_sim_step % self.sim_steps_per_display == 0:
                self.display()

        # Calculate the reward based on the position of the pendulum.
        self.rewards = [1.0 - np.cos(self.position)]

        self.smoothed_reward = (
            1 - self.reward_smoothing
        ) * self.smoothed_reward + self.reward_smoothing * self.rewards[0]

        self.step_sensors()

    def step_sensors(self):
        self.sensors = np.array([self.position, self.velocity])

    def display(self):
        n_lines = 4

        print(
            f"reward {self.smoothed_reward:.3}  at step {self.i_step:,},"
            + f" episode {self.i_episode}                                        "
        )

        # Build a string showing the current angle of the pendulum.
        n_angles = 72
        i_angle = 1 + int(n_angles * self.position / (2 * np.pi))
        position_list = ["_"] * (n_angles +  1)
        position_list[i_angle] = "0"
        position_string = "".join(position_list)
        print(position_string, "             ")
        print(
            "0        45       90      120     -180-     225      270      315      360"
        )
        angle = int(self.position * 360 / (2 * np.pi))
        angular_vel = int(self.velocity * 360 / (2 * np.pi))

        print("", flush=True)

        # Go to the beginning of the previous nth line.
        for _ in range(n_lines):
            print("\033[F", end="")
