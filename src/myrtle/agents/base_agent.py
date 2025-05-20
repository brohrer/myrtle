"""
Chooses a single random action at each step.
"""

import json
import time
import numpy as np
import dsmq.client
from myrtle.config import mq_host, mq_port

# How long to wait in between attempts to read from the message queue.
# For now this is hard coded.
# Less delay than this starts to bog down the mq server.
# More delay than this can result in a performance hit--a slight
# latency increase in the world -> agent communication.
_polling_delay = 0.001  # seconds


class BaseAgent:
    name = "Base agent"

    def __init__(self, **kwargs):
        self.init_common(**kwargs)

    def init_common(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        q_action=None,
        q_reward=None,
        q_sensor=None,
    ):
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards

        self.q_action = q_action
        self.q_reward = q_reward
        self.q_sensor = q_sensor

        # Initialize the mq as part of `run()` because it allows
        # process "spawn" method process forking to work, allowing
        # this code to run on macOS in addition to Linux.
        self.mq_initialized = False

    def initialize_mq(self):
        if not self.mq_initialized:
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
                self.receive_sensors_timestamp = 0
                self.send_actions_timestamp = 0
                step_loop_complete = False
                # Polling loop, waiting for new inputs
                while not (step_loop_complete or episode_complete or run_complete):
                    time.sleep(_polling_delay)

                    step_loop_complete = self.read_world_step()

                    # Each time through the polling loop, check
                    # whether the agent needs to be reset or terminated.
                    episode_complete, run_complete = self.control_check()

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
        sensor_success = False
        while not self.q_sensor.empty():
            self.sensors = self.q_sensor.get_nowait()
            sensor_success = True
            self.receive_sensors_timestamp = time.time()

        while not self.q_reward.empty():
            self.rewards = self.q_reward.get_nowait()

        return sensor_success

    def write_agent_step(self):
        self.q_action.put(self.actions)
        self.send_actions_timestamp = time.time()

        msg = json.dumps(
            {
                "step": self.i_step,
                "episode": self.i_episode,
                "actions": self.actions.tolist(),
                "ts_recv": int(1e6 * self.receive_sensors_timestamp),
                "ts_send": int(1e6 * self.send_actions_timestamp),
            }
        )
        self.mq.put("agent_step", msg)

    def control_check(self):
        episode_complete = False
        run_complete = False
        msg = self.mq.get("control")
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
            self.mq.close()
        except AttributeError:
            pass

        # Close down the Queue that the agent feeds
        self.q_action.close()
        self.q_action.cancel_join_thread()
