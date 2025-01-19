import json
import sys
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
        self.world_steps_per_loop_step = int(np.round(
            world_steps_per_second / loop_steps_per_second
        ))
        self.loop_steps_per_second = float(loop_steps_per_second)
        self.world_steps_per_second = float(
            self.world_steps_per_loop_step *
            self.loop_steps_per_second
        )
        self.loop_period = 1 / self.loop_steps_per_second
        self.world_period = 1 / self.world_steps_per_second

        self.pm = Pacemaker(self.world_steps_per_second)

        # Initialize message queue connection.
        self.mq = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)

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
        self.i_episode = 0
        while self.i_episode < self.n_episodes:
            self.i_episode += 1

            # `i_loop_step` counts the number of world->agent->world loop iterations,
            # time steps for the RL algo.
            # `i_world_step` counts the number of time steps internal to the world.
            # These can be much finer, as in the case of a physics simulation or
            # rapidly sampled sensors whose readings are aggregrated before passing
            # them on to the world.
            self.i_loop_step = 0
            self.i_world_step = 0

            self.reset()

            while self.i_loop_step < self.n_loop_steps:
                for _ in range(self.world_steps_per_loop_step):
                    self.i_world_step += 1
                    self.pm.beat()

                    # Trying to read agent action commands on every world step
                    # will allow the actions to
                    # start having an effect *almost* instantaneously.
                    # This is an approximate solution to the challenge of
                    # an agent taking non-negligible wall clock time to execute.
                    # There's more detail here:
                    # https://www.brandonrohrer.com/rl_noninteger_delay.html
                    # debug
                    print(self.i_world_step, self.i_loop_step, self.i_episode)
                    self.read_agent_step()
                    self.step_world()

                self.sense()
                self.write_world_step()
                self.i_loop_step += 1

            # Get ready to start the next episode
            self.mq.put("control", "truncated")

        # Wrap up the run
        self.mq.put("control", "terminated")
        sys.exit()

    def reset(self):
        """
        Re-initialize the world to its starting condition.

        Extend this with any other initializations, non-zero initial conditions,
        or physical actions that need to be taken.
        """
        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards

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

    def step_world(self):
        """
        One step of the (possibly much faster) hardware loop.

        Extend this class and implement your own step_world()
        """
        try:
            self.i_action = np.where(self.actions)[0][0]
        except IndexError:
            self.i_action = 1

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
            print(agent_msg)
            self.actions = json.loads(agent_msg)["actions"]

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

    """
    def initialize_log(self, log_name, log_dir, logging_level):
        if log_name is not None:
            # Create the columns and empty row data
            cols = []
            for i in range(self.n_sensors):
                cols.append(f"sen{i}")
            for i in range(self.n_actions):
                cols.append(f"act{i}")
            for i in range(self.n_rewards):
                cols.append(f"rew{i}")
            cols.append("loop_step")
            cols.append("episode")
            cols.append("timestamp")
            cols.append("note")

            # If the logger already exists, clean it out.
            try:
                old_logger = logging.open_logger(
                    name=log_name,
                    dir_name=log_dir,
                )
                old_logger.delete()
            except RuntimeError:
                pass

            self.logger = logging.create_logger(
                name=log_name,
                dir_name=log_dir,
                level=logging_level,
                columns=cols,
            )
        else:
            self.logger = None

    def log_step(self):
        if self.logger is not None:
            self.log_data = {}
            for i in range(self.n_sensors):
                self.log_data[f"sen{i}"] = self.sensors[i]
            for i in range(self.n_actions):
                self.log_data[f"act{i}"] = self.actions[i]
            for i in range(self.n_rewards):
                self.log_data[f"rew{i}"] = self.rewards[i]
            self.log_data["loop_step"] = self.i_loop_step
            self.log_data["episode"] = self.i_episode
            self.log_data["timestamp"] = time.time()

            self.logger.info(self.log_data)
    """

    """
    def close(self):
        ## self.sensor_q.put({"terminated": True})
        ## self.report_q.put({"terminated": True})

        # If a logger was created, delete it.
        # try:
        #     self.logger.delete()
        # except AttributeError:
        #     pass

        sys.exit()
    """
