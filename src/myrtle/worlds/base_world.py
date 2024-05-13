import sqlite3
import time
import numpy as np
from pacemaker.pacemaker import Pacemaker
from sqlogging import logging

N_SENSORS = 13
N_ACTIONS = 5
N_REWARDS = 3
STEPS_PER_SECOND = 5
N_TIME_STEPS = 93  # Number of time steps to run in a single episode
N_EPISODES = 4


class BaseWorld:
    def __init__(
            self,
            sensor_q=None,
            action_q=None,
            log_dir=".",
            logging_level="info",
    ):
        self.n_sensors = N_SENSORS
        self.n_actions = N_ACTIONS
        self.n_rewards = N_REWARDS
        self.sensor_q = sensor_q
        self.action_q = action_q

        self.pm = Pacemaker(STEPS_PER_SECOND)
        self.initialize_log(log_name, log_dir, logging_level)

        self.i_episode = -1
        self.reset()

    def reset(self):
        self.sensors = np.zeros(self.n_sensors)
        self.actions = np.zeros(self.n_actions)
        self.rewards = [0] * self.n_rewards
        self.i_step = 0
        self.i_episode += 1

    def run(self):
        while self.i_episode < N_EPISODES:
            while self.i_step < N_TIME_STEPS:

                # Read q
                # If all goes well, there should be exactly one action
                # array in the queue.
                # If multiple, use the last and ignore others.
                # If none, use an all-zeros action. 
                self.actions = np.zeros(self.n_actions)
                while not action_q.is_empty():
                    msg = action_q.get()
                    self.actions = msg["actions"]

                try:
                    self.sensors = msg["sensors"]
                except KeyError:
                    pass
                try:
                    self.rewards = msg["rewards"]
                except KeyError:
                    pass

                self.step()
                self.action_q.put({"actions": self.actions})


            self.action_q.put({"truncated": True})
            self.reset()

        self.action_q.put({"terminated": True})
        self.close()

    def step(self):
        self.i_step += 1
        # Pick a random action.
        self.actions = np.zeros(self.n_actions)
        i_action = np.random.choice(self.n_actions)
        self.actions[i_action] = 1

        self.log_step()

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
            except sqlite3.OperationalError:
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
        self.logger.delete()
