# import json
import os

# import time
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.colors import LinearSegmentedColormap
from myrtle.config import log_directory
import matplotlib.patches as patches

# import dsmq.client
# from myrtle.agents.tools import retrieve_buckettree_info
# from myrtle.config import mq_host, mq_port
import myrtle.reports.report_config as config

history_length_short = 5
history_length_med = 20
history_length_long = 100
history_length_super_long = 1000

range_buffer = 0.08

_connection_wait = 2.0  # seconds
_n_tries = 10


def report_buckettree():
    n_trees = 0
    highs = []
    lows = []
    levels = []

    # Read in all the tree data from the tree info directories
    for dirname in os.listdir(os.path.join(log_directory, "buckettree")):
        if dirname[:5] == "tree_":
            n_trees += 1
            highs.append(
                np.load(os.path.join(log_directory, "buckettree", dirname, "highs.npy"))
            )
            lows.append(
                np.load(os.path.join(log_directory, "buckettree", dirname, "lows.npy"))
            )
            levels.append(
                np.load(
                    os.path.join(log_directory, "buckettree", dirname, "levels.npy")
                )
            )

    fig, axes_list = config.blank_images(n_trees)

    for i_ax in range(n_trees):
        ax = axes_list[-i_ax - 1]
        draw_buckets(ax, highs[i_ax], lows[i_ax], levels[i_ax])
        ax.set_xlabel(
            f"sensor {i_ax}",
            color=config.color,
            fontsize=config.fontsize_small,
        )
    plt.show()


def draw_buckets(ax, highs, lows, levels):
    min_lo = 1e10
    max_hi = -1e10
    max_level = -1
    for i, hi in enumerate(highs):
        lo = lows[i]
        lv = levels[i]

        if lo < min_lo:
            if lo > -1e10:
                min_lo = lo
            else:
                lo = -1e10

        if hi > max_hi:
            if hi < 1e10:
                max_hi = hi
            else:
                hi = 1e10

        if lv > max_level:
            max_level = lv

        path = [
            [lo, -lv - 0.5],
            [hi, -lv - 0.5],
            [hi, -lv + 0.5],
            [lo, -lv + 0.5],
        ]
        ax.add_patch(
            patches.Polygon(
                path,
                alpha=0.2,
                edgecolor=config.color,
                facecolor=config.color,
                joinstyle="miter",
                linewidth=config.linewidth_med,
            )
        )

    if max_hi < min_lo:
        xmin = -1
        xmax = 1
    elif max_hi == min_lo:
        xmin = min_lo - 1
        xmax = max_hi + 1
    else:
        x_range = max_hi - min_lo
        xmin = min_lo - x_range * range_buffer
        xmax = max_hi + x_range * range_buffer
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-(max_level + 1), 1.0)


def left_title(ax, text):
    xmin, xmax = ax.get_xlim()
    x_text = xmin - (xmax - xmin) * 0.13
    ax.text(
        x_text,
        ax.get_ylim()[1],
        text,
        horizontalalignment="left",
        verticalalignment="bottom",
        color=config.color,
        fontsize=config.fontsize_small,
    )


if __name__ == "__main__":
    report_buckettree()
