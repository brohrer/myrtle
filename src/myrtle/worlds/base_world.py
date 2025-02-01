import json
import numpy as np
import dsmq.client
from myrtle import config
from pacemaker.pacemaker import Pacemaker

_default_n_loop_steps = 101
_default_n_episodes = 3
_default_loop_steps_per_second = 5.0
_default_speedup = 1.0


class BaseWorld:
    """
    Extend this class to make your own World.

    It is designed so that you'll (ideally) only need to override
    * `__init__()`
    * `reset()`
    * `step_world()`
    * `sense()`

    but you may find you need to dig deeper to get the behaviors you want.
    """

    def __init__(
        self,
        n_loop_steps=100,
        n_episodes=1,
        loop_steps_per_second=10.0,
        world_steps_per_second=None,
        speedup=1.0,
    ):
        """
        When extending `BaseWorld` to cover a new world, override `__init__()`
        with the constants and settings of the new world.

        Unless you have a good reason not to, include a call to `init_common()`
        at the beginning and a call to `reset()` at the end, as shown here.

        `world_steps_per_second` should be an integer multiple of
        `loop_steps_per_second` so that there are the same number of world steps
        in each loop step. If it's not, `world_steps_per_second`
        will be rounded to the nearest available integer option.
        """
        # Take care of the myrtle-specific boilerplate stuff.
        # It initializes the wall clock time keeper (pacemaker) and the
        # shared communication channels (dsmq).
        self.init_common(
            n_loop_steps=n_loop_steps,
            n_episodes=n_episodes,
            loop_steps_per_second=loop_steps_per_second,
            world_steps_per_second=world_steps_per_second,
            speedup=speedup,
        )
        self.name = "Base world"

        self.n_sensors = 13
        self.n_actions = 5
        self.n_rewards = 3

    def init_common(
        self,
        n_loop_steps=_default_n_loop_steps,
        n_episodes=_default_n_episodes,
        loop_steps_per_second=_default_loop_steps_per_second,
        world_steps_per_second=None,
        speedup=_default_speedup,
    ):
        """
        This boilerplate will need to be run when initializing most worlds.
        """
        self.n_loop_steps = n_loop_steps
        self.n_episodes = n_episodes

        # Default to one world step per loop step
        if world_steps_per_second is None:
            world_steps_per_second = loop_steps_per_second

        # The world will run at one clock rate, and the interaction loop
        # with the agent will run at another. For convenience and repeatability,
        # ensure that the loop interaction steps contain
        # a consistent number of world time steps.
        self.world_steps_per_loop_step = int(
            np.round(world_steps_per_second / loop_steps_per_second)
        )
        self.loop_steps_per_second = float(loop_steps_per_second)
        self.world_steps_per_second = float(
            self.world_steps_per_loop_step * self.loop_steps_per_second
        )
        self.loop_period = 1 / self.loop_steps_per_second
        self.world_period = 1 / self.world_steps_per_second

        self.pm = Pacemaker(self.world_steps_per_second)

        # Initialize the mq as part of `run()` because it allows
        # process "spawn" method process foring to work, allowing
        # this code to run on macOS in addition to Linux.
        self.mq_initialized = False

    def initialize_mq(self):
        if not self.mq_initialized:
            # Initialize message queue socket.
            self.mq = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)
            self.mq_initialized = True

    def run(self):
        """
        This is the entry point for setting a world in motion.
        ```python
        world = BaseWorld()
        world.run()
        ```

        If all goes as intended, you won't need to modify this method.
        Of course we both know things never go precisely as intended.
        """
        self.initialize_mq()
        for i_episode in range(self.n_episodes):
            self.i_episode = i_episode

            # `i_loop_step` counts the number of world->agent->world loop iterations,
            # time steps for the RL algo.
            # `i_world_step` counts the number of time steps internal to the world.
            # These can be much finer, as in the case of a physics simulation or
            # rapidly sampled sensors whose readings are aggregrated before passing
            # them on to the world.
            # self.i_loop_step = -1
            # self.i_world_step = -1

            self.reset()

            for i_loop_step in range(self.n_loop_steps):
                self.i_loop_step = i_loop_step
                print(
                    f"episode {self.i_episode}  loop step {self.i_loop_step}", end="\r"
                )

                for i_world_step in range(self.world_steps_per_loop_step):
                    self.i_world_step = i_world_step
                    self.pm.beat()

                    # Trying to read agent action commands on every world step
                    # will allow the actions to
                    # start having an effect *almost* instantaneously.
                    # This is an approximate solution to the challenge of
                    # an agent taking non-negligible wall clock time to execute.
                    # There's more detail here:
                    # https://www.brandonrohrer.com/rl_noninteger_delay.html
                    self.read_agent_step()
                    self.step_world()

                self.sense()
                self.write_world_step()
                time_to_shutdown = self.shutdown_check()
                if time_to_shutdown:
                    break

            if time_to_shutdown:
                break
            # Get ready to start the next episode
            self.mq.put("control", "truncated")

        # Wrap up the run
        self.mq.put("control", "terminated")
        self.close()

    def reset(self):
        """
        Re-initialize the world to its starting condition.

        Extend this with any other initializations, non-zero initial conditions,
        or physical actions that need to be taken.
        """
        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

    def read_agent_step(self):
        # Read in any actions that the agent has put in the message queue.
        # In a syncronous world-agent loop, there should be exactly one action
        # array in the queue.
        #
        # If there is one action command report it.
        # If there are multiple, report the last and ignore the others.
        # If there are none, report an all-zeros action.
        self.actions = np.zeros(self.n_actions)
        while True:
            agent_msg = self.mq.get("agent_step")
            if agent_msg == "":
                break
            self.actions = np.array(json.loads(agent_msg)["actions"])

    def step_world(self):
        """
        One step of the (possibly much faster) hardware loop.
        Evolve the internal state of the world over the loop time step.

        Extend this class and implement your own step_world()
        """
        try:
            self.i_action = np.where(self.actions)[0][0]
        except IndexError:
            self.i_action = 1

    def sense(self):
        """
        One step of the sense -> act -> reward RL loop.

        Extend this class and implement your own sense()
        """

        # Some arbitrary, but deterministic behavior.
        self.sensors = np.zeros(self.n_sensors)
        self.sensors[: self.n_actions] = self.actions
        self.sensors[self.n_actions : 2 * self.n_actions] = 0.8 * self.actions - 0.3

        self.rewards = [0] * self.n_rewards
        self.rewards[0] = self.i_action / 10
        self.rewards[1] = -self.i_action / 2
        self.rewards[2] = self.i_action / (self.i_loop_step + 1)
        if self.i_action < self.n_rewards:
            self.rewards[self.i_action] = None

    def write_world_step(self):
        msg = json.dumps(
            {
                "loop_step": self.i_loop_step,
                "episode": self.i_episode,
                "sensors": self.sensors.tolist(),
                "rewards": self.rewards,
            }
        )
        self.mq.put("world_step", msg)

    def shutdown_check(self):
        # Check whether there has been at "terminated" control message
        # issued from the workbench process.
        time_to_shutdown = False
        msg = self.mq.get("control")
        if msg in ["terminated", "shutdown"]:
            time_to_shutdown = True
        return time_to_shutdown

    def close(self):
        # If self.mq has been initialized, close it down.
        try:
            self.mq.close()
        except AttributeError:
            pass
