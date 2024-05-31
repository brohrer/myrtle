import sys
import time
import numpy as np
from sqlogging import logging


class BaseAgent:
    def __init__(
        self,
        n_sensors=None,
        n_actions=None,
        n_rewards=None,
        sensor_q=None,
        action_q=None,
        log_name=None,
        log_dir=".",
        logging_level="info",
    ):
        self.name = "Base agent"
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.n_rewards = n_rewards
        self.sensor_q = sensor_q
        self.action_q = action_q

        self.initialize_log(log_name, log_dir, logging_level)

        # This will get incremented to 0 by the reset.
        self.i_episode = -1
        self.reset()

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.rewards = [0] * self.n_rewards
        self.actions = np.zeros(self.n_actions)
        self.i_episode += 1
        self.i_step = 0

    def step(self):
        # Pick a random action.
        self.actions = np.zeros(self.n_actions)
        i_action = np.random.choice(self.n_actions)
        self.actions[i_action] = 1

    def run(self):
        while True:
            # It's possible that there may be more than one batch of sensor
            # information. The agent will have to handle that case.
            # Wait around until there is at least one batch.
            msg = self.sensor_q.get()
            # If the agent needs to be reset or shut down, handle that.
            try:
                if msg["terminated"]:
                    self.close()
            except KeyError:
                pass
            try:
                if msg["truncated"]:
                    self.reset()
            except KeyError:
                pass

            # Then check to see if there are are any others.
            while not self.sensor_q.empty():
                msg = self.sensor_q.get()
                # It's important to check for termination or truncation in every
                # message. Missing one of them is bad news.
                try:
                    if msg["terminated"]:
                        self.close()
                except KeyError:
                    pass
                try:
                    if msg["truncated"]:
                        self.reset()
                except KeyError:
                    pass

            try:
                self.sensors = msg["sensors"]
            except KeyError:
                pass
            try:
                self.rewards = msg["rewards"]
            except KeyError:
                pass

            self.step()
            self.log_step()
            self.i_step += 1

            self.action_q.put({"actions": self.actions})

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
            cols.append("i_step")
            cols.append("i_episode")
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
            self.log_data["i_step"] = self.i_step
            self.log_data["i_episode"] = self.i_episode
            self.log_data["timestamp"] = time.time()

            self.logger.info(self.log_data)

    def close(self):
        # If a logger was created, delete it.
        try:
            self.logger.delete()
        except AttributeError:
            pass

        sys.exit()
