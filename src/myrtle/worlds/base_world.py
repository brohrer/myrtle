import sqlite3
import sys
import time
import numpy as np
from pacemaker.pacemaker import Pacemaker
from sqlogging import logging

N_SENSORS = 13
N_ACTIONS = 5
N_REWARDS = 3
STEPS_PER_SECOND = 5
N_TIME_STEPS = 11  # Number of time steps to run in a single episode
N_EPISODES = 2


class BaseWorld:
    def __init__(
            self,
            sensor_q=None,
            action_q=None,
            report_q=None,
            log_name=None,
            log_dir=".",
            logging_level="info",
    ):
        self.name = "Base world"
        self.n_sensors = N_SENSORS
        self.n_actions = N_ACTIONS
        self.n_rewards = N_REWARDS
        self.sensor_q = sensor_q
        self.action_q = action_q
        self.report_q = report_q

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
        self.sensor_q.put({"truncated": True})

    def run(self):
        while self.i_episode < N_EPISODES:
            while self.i_step < N_TIME_STEPS:
                self.pm.beat()

                # Read q
                # If all goes well, there should be exactly one action
                # array in the queue.
                # If multiple, use the last and ignore others.
                # If none, use an all-zeros action. 
                self.actions = np.zeros(self.n_actions)
                while not self.action_q.empty():
                    msg = self.action_q.get()
                    self.actions = msg["actions"]

                self.step()
                self.sensor_q.put({
                    "sensors": self.sensors,
                    "rewards": self.rewards,
                })
                self.report_q.put({
                    "step": self.i_step,
                    "episode": self.i_episode,
                    "rewards": self.rewards,
                })

            self.reset()

        self.close()

    def step(self):
        self.i_step += 1

        try:
            i_action = np.where(self.actions)[0][0]
        except IndexError:
            i_action = 1

        # Some arbitrary, but deterministic behavior.
        self.sensors  = np.zeros(self.n_sensors)
        self.sensors[:self.n_actions] = self.actions
        self.sensors[self.n_actions: 2 * self.n_actions] = .8 * self.actions - .3

        self.rewards = [0] * self.n_rewards
        self.rewards[0] = i_action / 10
        self.rewards[1] = -i_action / 2
        self.rewards[2] = i_action / self.i_step
        if i_action < self.n_rewards:
            self.rewards[i_action] = None

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
            cols.append("step")
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
            self.log_data["step"] = self.i_step
            self.log_data["episode"] = self.i_episode
            self.log_data["timestamp"] = time.time()

            self.logger.info(self.log_data)

    def close(self):
        self.sensor_q.put({"terminated": True})

        # If a logger was created, delete it.
        try:
            self.logger.delete()
        except AttributeError:
            pass

        sys.exit()
