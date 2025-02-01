import os
import time

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.patches as patches
# from matplotlib.ticker import FixedLocator, FixedFormatter
import dsmq.client
from myrtle import config

from pacemaker.pacemaker import Pacemaker

plt.rcParams["toolbar"] = "None"

CLOCK_FREQ = 24.0  # Hz
TIMEOUT = 2.0  # seconds
STEP_HISTORY_LENGTH = 120
EPISODE_HISTORY_LENGTH = 1000

WINDOW_TITLE = "Workbench Dash"
BACKGROUND_COLOR = "#000000"
FOREGROUND_COLOR = "#BBBBBB"
SECOND_COLOR = "#666666"
BORDER_VERT = 0.12
BORDER_HORIZ = 0.08
STEP_LINE_PARAMS = {
    "color": FOREGROUND_COLOR,
    "linewidth": 1.0,
    # "marker": "o",
    # "markersize": 3,
}
EPISODE_LINE_PARAMS = {
    "color": FOREGROUND_COLOR,
    "linewidth": 0.0,
    "marker": "o",
    "markersize": 2,
}

X_LABEL_REWARD = "step"
Y_LABEL = "reward"
X_LABEL_EPISODE = "episode"
LABEL_PARAMS = {
    "color": SECOND_COLOR,
    "fontsize": 9,
}
"""
X_TICK_LABELS = ["60", "45", "30", "15", "now"]
Y_TICK_LABELS = ["50%", "100%", "150%", "200%"]
X_TICK_POSITIONS = [-59, -45, -30, -15, 0]
Y_TICK_POSITIONS = [0.5, 1.0, 1.5, 2.0]
"""
GRID_PARAMS = {
    "color": SECOND_COLOR,
    "linewidth": 1,
    "linestyle": "dotted",
}
X_TICK_PARAMS = {
    "axis": "x",
    "direction": "in",
    "color": SECOND_COLOR,
    "labelcolor": SECOND_COLOR,
    "labelsize": 7,
    "bottom": False,
    "top": False,
    "labelbottom": True,
    "labeltop": False,
}
Y_TICK_PARAMS = {
    "axis": "y",
    "direction": "in",
    "color": SECOND_COLOR,
    "labelcolor": SECOND_COLOR,
    "labelsize": 7,
    "left": False,
    "right": False,
    "labelleft": True,
    "labelright": False,
}
"""
CONCERN_ZONE_PARAMS = {
    "edgecolor": "none",
    "facecolor": "#333333",
    "zorder": -2,
}
"""


# def run(dash_q, window_pixels):
def run(window_pixels):
    frame = Frame(window_pixels=window_pixels)
    pacemaker = Pacemaker(CLOCK_FREQ)
    last_observation_timestamp = time.time()

    mq = dsmq.client.connect(config.MQ_HOST, config.MQ_PORT)
    while True:
        pacemaker.beat()

        # Empty out whatever observations are currently in the queue.
        while True:
            # Check whether there is new reward value reported.
            msg_str = mq.get("world_step")
            if msg_str == "":
                mq_empty = True
                break
            else:
                msg = json.loads(msg_str)
                frame.add_observation(msg)
                last_observation_timestamp = time.time()

        if (time.time() - last_observation_timestamp) > TIMEOUT:
            print("Workbench dashboard has timed out. Shutting it down.")
            os._exit(os.EX_OK)

        frame.update()


class Frame:
    def __init__(self, window_pixels):
        self.reward_history = np.zeros(STEP_HISTORY_LENGTH)
        self.episode_history = -1e100 * np.ones(EPISODE_HISTORY_LENGTH)

        total_reward = 0.0
        total_steps = 0

        self.i_episode = 0
        self.episode_stepwise_history = []

        self.fig = plt.figure(
            facecolor=BACKGROUND_COLOR,
            num=WINDOW_TITLE,
        )

        # Set the window location for the dashboard.
        try:
            # My machine uses backend QtAgg.
            # This approach won't work for other backends.
            # To find out which backend you are using
            # uncomment this snippet.
            # import matplotlib
            # print(matplotlib.get_backend())
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            x, y, width, height = window_pixels
            mngr.window.setGeometry(x, y, width, height)
        except Exception:
            # If unsuccessful, don't worry about it.
            pass

        ax_width = (1 - 3 * BORDER_HORIZ) / 2
        ax_height = 1 - 2 * BORDER_VERT
        self.step_reward_ax = self.add_dash_axes(
            (BORDER_HORIZ, BORDER_VERT, ax_width, ax_height)
        )

        steps = np.arange(-len(self.reward_history), 0)
        self.step_reward_line = self.step_reward_ax.plot(
            steps, self.reward_history, **STEP_LINE_PARAMS
        )[0]
        self.step_reward_ax.set_xlabel(X_LABEL_REWARD, **LABEL_PARAMS)
        self.step_reward_ax.set_ylabel(Y_LABEL, **LABEL_PARAMS)

        ax_left = ax_width + 2 * BORDER_HORIZ
        self.episode_reward_ax = self.add_dash_axes(
            (ax_left, BORDER_VERT, ax_width, ax_height)
        )
        episodes = np.arange(-len(self.episode_history), 0)
        self.episode_reward_line = self.episode_reward_ax.plot(
            episodes, self.episode_history, **EPISODE_LINE_PARAMS
        )[0]
        self.episode_reward_ax.set_xlabel(X_LABEL_EPISODE, **LABEL_PARAMS)
        self.episode_reward_ax.tick_params(labelleft=False)

        plt.ion()
        plt.show()

    def add_observation(self, observation):
        # observation = {"rewards":..., "step":..., "episode":...}
        reward = 0.0
        try:
            for reward_channel in observation["rewards"]:
                if reward_channel is not None:
                    reward += reward_channel
        except KeyError:
            # Rewards not yet populated.
            pass

        step = observation["step"]
        episode = observation["episode"]

        total_reward += reward
        total_steps += 1
        show_report = False

        if episode > self.i_episode:
            episode_mean_reward = np.mean(self.episode_stepwise_history)
            self.episode_stepwise_history = []
            self.i_episode += 1
            show_report = True

            self.episode_history[0] = episode_mean_reward
            self.episode_history = np.roll(self.episode_history, -1)

        self.episode_stepwise_history.append(reward)

        self.reward_history[0] = reward
        self.reward_history = np.roll(self.reward_history, -1)

        if show_report:
            avg_reward = total_reward / total_steps
            print()
            if episode > 1:
                print(
                    f"Lifetime average reward across {episode} episodes"
                    + f" of {step} steps each"
                )
                print(f"for {agent.name} on {world.name}: {avg_reward}")
            else:
                print(
                    f"    Lifetime average reward for {agent.name}"
                    + f" on {world.name}: {avg_reward}"
                )

    def update(self):
        self.step_reward_line.set_ydata(self.reward_history)

        episodes = np.arange(
            -len(self.episode_history) + self.i_episode, self.i_episode
        )
        self.episode_reward_line.set_xdata(episodes)
        self.episode_reward_line.set_ydata(self.episode_history)
        episode_x_min = np.maximum(-1, np.min(episodes) - 1)
        episode_x_max = np.max(episodes) + 1
        self.episode_reward_ax.set_xlim(episode_x_min, episode_x_max)

        r_min = np.min(self.reward_history)
        r_max = np.max(self.reward_history)
        r_range = r_max - r_min
        y_min, y_max = self.step_reward_ax.get_ylim()
        self.step_reward_ax.set_ylim(
            np.minimum(y_min, r_min - r_range * 0.05),
            np.maximum(y_max, r_max + r_range * 0.05),
        )
        self.episode_reward_ax.set_ylim(self.step_reward_ax.get_ylim())
        self.fig.canvas.flush_events()

    def add_dash_axes(self, extents):
        ax = self.fig.add_axes(extents)

        ax.set_facecolor(BACKGROUND_COLOR)

        """
        # Do some customization
        x_formatter = FixedFormatter(X_TICK_LABELS)
        y_formatter = FixedFormatter(Y_TICK_LABELS)
        x_locator = FixedLocator(X_TICK_POSITIONS)
        y_locator = FixedLocator(Y_TICK_POSITIONS)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        """

        ax.tick_params(**X_TICK_PARAMS)
        ax.tick_params(**Y_TICK_PARAMS)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(SECOND_COLOR)

        # ax.grid(**GRID_PARAMS)
        """
        ax.plot([t_min, t_max], [2.0, 2.0], **GRID_PARAMS)
        ax.plot([t_min, t_max], [1.5, 1.5], **GRID_PARAMS)
        ax.plot([t_min, t_max], [1.0, 1.0], **GRID_PARAMS)
        ax.plot([t_min, t_max], [0.5, 0.5], **GRID_PARAMS)
        """
        """
        # Create a shaded patch showing the zone of concern
        path = [
            [t_min, 1.0],
            [t_min, 10.0],
            [t_max, 10.0],
            [t_max, 1.0],
        ]
        ax.add_patch(patches.Polygon(path, **CONCERN_ZONE_PARAMS))
        """

        return ax
