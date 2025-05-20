import json
import numpy as np
from myrtle.worlds.base_world import BaseWorld
from myrtle.config import monitor_host, monitor_port
from myrtle.worlds.tools.ring_buffer import RingBuffer

_default_world_steps_per_loop_step = 8


class Pendulum(BaseWorld):
    """
    A pendulum that starts at rest. Reward comes from keeping the pendulum
    elevated. Inverting it is optimal.

    This World extends the BaseWorld, which tries to take care of the overhead
    and bookkeeping associated with being part of the Myrtle framework.
    Invariably, there will be some part of the under-the-hood implementation
    that you will need to read or customize. Refer to the `base_world.py`
    code as the authoritative source.
    https://codeberg.org/brohrer/myrtle/src/branch/main/src/myrtle/worlds/base_world.py

    Position convention
        0 radians is straight down,
        pi / 2 radians is to the right
        pi radians is stratight up

    Action convention
        Positive actions are counter-clockwise torque.
        Negative actions are clockwise torque.
        All torques are in Newton-meters.
    """

    name = "Pendulum"

    def __init__(
        self,
        n_loop_steps=1000,
        n_episodes=1,
        loop_steps_per_second=4,
        world_steps_per_second=None,
        speedup=8,
        verbose=True,
        **kwargs,
    ):
        if world_steps_per_second is None:
            world_steps_per_second = (
                loop_steps_per_second * _default_world_steps_per_loop_step
            )
        self.init_common(
            n_loop_steps=n_loop_steps,
            n_episodes=n_episodes,
            loop_steps_per_second=loop_steps_per_second,
            world_steps_per_second=world_steps_per_second,
            speedup=speedup,
            verbose=verbose,
            **kwargs,
        )
        print(f"""
    Watch the pendulum swing:  http://{monitor_host}:{monitor_port}/pendulum.html
        """)

        self.reset()
        self.action_scale = 16 * np.array(
            [
                -1.0,
                -0.75,
                -0.5,
                -0.375,
                -0.25,
                -0.125,
                0.0,
                0.125,
                0.25,
                0.375,
                0.5,
                0.75,
                1.0,
            ]
        )
        self.n_actions = self.action_scale.size
        self.n_rewards = 1

        vis_updates_per_second = 60
        self.world_steps_per_vis_update = int(
            np.ceil(self.world_steps_per_second / vis_updates_per_second)
        )

        self.mass = 1  # kilogram
        self.length = 2  # meter
        self.inertia = self.mass * self.length**2 / 12  # rotational inertia units
        self.gravity = -9.8  # meters / second^2
        self.friction = -0.30  # Newton-meters-seconds / radian

        self.dt = 1.0 / self.world_steps_per_second

        impulse_length = int(world_steps_per_second / loop_steps_per_second)
        self.impulse = np.ones(impulse_length)

    def reset(self):
        self.position = 0  # radians
        self.velocity = 0  # radians per second
        self.torque_buffer = RingBuffer(self.world_steps_per_loop_step)

        self.reset_sensors()

    def reset_sensors(self):
        self.n_sensors = 2

        self.sensors = np.array([self.position, self.velocity])

    def sense(self):
        # Calculate the reward based on the position of the pendulum.
        self.rewards = [1.0 - np.cos(self.position)]

        self.step_sensors()

    def step_world(self):
        # Add any new actions to the torque buffer.
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
        new_position = self.position + self.velocity * self.dt

        # Keep position in the range of [0, 2 pi)
        self.position = np.mod(new_position, 2 * np.pi)

    def step_sensors(self):
        self.sensors = np.array([self.position, self.velocity])
        self.write_pendulum_state()

    def write_pendulum_state(self):
        msg = json.dumps(
            {
                "loop_step": self.i_loop_step,
                "episode": self.i_episode,
                "position": self.position,
                "velocity": self.velocity,
            }
        )
        self.mq.put("pendulum_state", msg)
