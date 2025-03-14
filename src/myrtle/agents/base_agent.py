"""
Chooses a single random action at each step.
"""

import json
import time
import numpy as np
import dsmq.client
from myrtle.config import mq_loop_host, mq_loop_port, mq_host, mq_port

# How long to wait in between attempts to read from the message queue.
# For now this is hard coded.
# Less delay than this starts to bog down the mq server.
# More delay than this can result in a performance hit--a slight
# latency increase in the world -> agent communication.
_polling_delay = 0.002  # seconds


class BaseAgent:
    name = "Base agent"

    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
    ):
        self.init_common(
            n_sensors=n_sensors,
            n_actions=n_actions,
            n_rewards=n_rewards,
        )

    def init_common(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
    ):
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards

        # Initialize the mq as part of `run()` because it allows
        # process "spawn" method process foring to work, allowing
        # this code to run on macOS in addition to Linux.
        self.mq_initialized = False

    def initialize_mq(self):
        if not self.mq_initialized:
            # Initialize loop-specific message queue client.
            self.mq_loop = dsmq.client.connect(mq_loop_host, mq_loop_port)

            # Initialize general purpose message queue client.
            self.mq = dsmq.client.connect(mq_host, mq_port)

            self.mq_initialized = True

    def run(self):
        self.initialize_mq()
        run_complete = False
        self.i_episode = -1
        # Episode loop
        while not run_complete:
            self.i_episode += 1
            episode_complete = False
            self.i_step = -1
            self.reset()
            # world->agent->world step loop
            while not (episode_complete or run_complete):
                self.i_step += 1
                step_loop_complete = False
                # Polling loop, waiting for new inputs
                while not (step_loop_complete or episode_complete or run_complete):
                    time.sleep(_polling_delay)
                    step_loop_complete = self.read_world_step()
                    # Each time through the polling loop, check
                    # whether the agent needs to be reset or terminated.
                    episode_complete, run_complete = self.control_check()
                    self.write_agent_step()

                self.choose_action()
                self.write_agent_step()

        self.close()

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.rewards = [0] * self.n_rewards
        self.actions = np.zeros(self.n_actions)

    def choose_action(self):
        # Pick a random action.
        self.actions = np.zeros(self.n_actions)
        i_action = np.random.choice(self.n_actions)
        self.actions[i_action] = 1

    def read_world_step(self):
        # It's possible that there may be no sensor information available.
        # If not, just skip to the next iteration of the loop.
        response = self.mq_loop.get("world_step")
        if response == "":
            return False

        # It's possible that there may be more than one batch of sensor
        # information. If there is, skip to the latest batch.
        while response != "":
            msg_str = response
            response = self.mq_loop.get("world_step")

        msg = json.loads(msg_str)
        try:
            self.sensors = np.array(msg["sensors"])
        except KeyError:
            pass
        try:
            self.rewards = msg["rewards"]
        except KeyError:
            pass

        return True

    def write_agent_step(self):
        msg = json.dumps(
            {
                "actions": self.actions.tolist(),
                "step": self.i_step,
                "episode": self.i_episode,
                "timestamp": time.time(),
            }
        )
        self.mq_loop.put("agent_step", msg)
        self.mq.put("agent_step", msg)

    def control_check(self):
        episode_complete = False
        run_complete = False
        msg = self.mq_loop.get("control")
        if msg != "":
            # If this episode is over, begin the next one.
            if msg == "truncated":
                episode_complete = True
            # If the agent needs to be shut down, handle that.
            if msg == "terminated":
                run_complete = True
        return episode_complete, run_complete

    def close(self):
        # If mq clients have been initialized, close them down.
        try:
            self.mq_loop.close()
        except AttributeError:
            pass

        try:
            self.mq.close()
        except AttributeError:
            pass
