import argparse
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sqlogging import logging
from myrtle.config import log_directory
import myrtle.reports.report_config as rc

history_length_short = 5
history_length_med = 20
history_length_long = 100
history_length_super_long = 1000


def report_timing(db_name):
    n_plots = 8
    fig, axes_list = rc.blank_report(n_plots)
    # From bottom to top
    (
        ax_world,
        ax_handoff_world,
        ax_agent,
        ax_handoff_agent,
        ax_step,
        ax_long,
        ax_med,
        ax_short,
    ) = axes_list

    ax_world.set_xlabel("milliseconds", color=rc.color, fontsize=rc.fontsize_small)

    results = retrieve_timing(db_name, history_length_super_long)

    plot_timing(ax_short, results[: int(history_length_short * 2), :])
    plot_timing(ax_med, results[: int(history_length_med * 2), :])
    plot_timing(ax_long, results[: int(history_length_long * 2), :])

    results_reversed = results[::-1, :]
    process = results_reversed[:, 0]
    ts_recv = np.array(results_reversed[:, 3], float) / 1000  # convert to ms
    ts_send = np.array(results_reversed[:, 4], float) / 1000  # convert to ms

    world_count = 0
    agent_count = 0
    last_agent_send = None
    last_world_send = None
    step_duration = []
    world_duration = []
    agent_duration = []
    handoff_to_agent_duration = []
    handoff_to_world_duration = []

    for i, proc in enumerate(process):
        if proc == "world":
            world_count += 1
            world_duration.append(ts_send[i] - ts_recv[i])

            if last_world_send is not None:
                step_duration.append(ts_send[i] - last_world_send)

            if last_agent_send is not None:
                handoff_to_world_duration.append(ts_recv[i] - last_agent_send)

            last_world_send = ts_send[i]
            last_agent_send = None

        else:
            agent_count += 1
            agent_duration.append(ts_send[i] - ts_recv[i])

            if last_world_send is not None:
                handoff_to_agent_duration.append(ts_recv[i] - last_world_send)

            last_agent_send = ts_send[i]

    n_bins_approx = 100
    bin_width = int(100 * np.mean(step_duration) / n_bins_approx) / 100.0
    upper_bin = (np.ceil(np.max(step_duration) / bin_width) + 1) * bin_width
    bins = np.arange(0.0, upper_bin, bin_width)

    rc.plot_distribution(ax_step, step_duration, bins, "step total")
    rc.plot_distribution(
        ax_handoff_agent, handoff_to_agent_duration, bins, "handoff to agent"
    )
    rc.plot_distribution(ax_agent, agent_duration, bins, "agent step")
    rc.plot_distribution(
        ax_handoff_world, handoff_to_world_duration, bins, "handoff to world"
    )
    rc.plot_distribution(ax_world, world_duration, bins, "world step")

    agent_success_rate = 1 - (world_count - agent_count) / world_count
    rounded_agent_success_rate = int(1000 * (agent_success_rate)) / 10.0
    success_text = f"agent on-time completion rate is {rounded_agent_success_rate}%"

    rc.left_title(ax_short, success_text)

    report_filename = f"timing_{db_name}.png"
    plt.savefig(os.path.join(log_directory, report_filename))

    plt.show()


def retrieve_timing(db_name, n_steps=100):
    logger = logging.open_logger(
        name=db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT
            process,
            step,
            episode,
            ts_recv,
            ts_send
        FROM {db_name}
        ORDER BY ts_send DESC
        LIMIT {int(n_steps * 2)}
    """
    )
    results = np.array(result)
    return results


def convert_to_swimlane_axes(ax, n_lanes=None):
    ax.grid(False)
    ax.set_ylim(-0.5, n_lanes + 0.5)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)


def plot_timing(ax, results):
    process = results[:, 0]
    # step = np.array(results[:, 1], int)
    ts_recv = np.array(results[:, 3], float) / 1000  # convert to ms
    ts_send = np.array(results[:, 4], float) / 1000  # convert to ms
    current_time = np.max(ts_send)
    ts_recv -= current_time
    ts_send -= current_time

    convert_to_swimlane_axes(ax, n_lanes=2)
    ax.set_xlim(np.min(ts_recv[np.where(ts_recv > -1e9)]), 0)

    for i, proc in enumerate(process):
        x_end = ts_send[i]
        x_start = ts_recv[i]
        if x_end - x_start > 1e9:
            x_start = x_end

        if proc == "world":
            y_min = 1
            y_max = 2
        else:
            y_min = 0
            y_max = 1

        path = [
            [x_start, y_min],
            [x_end, y_min],
            [x_end, y_max],
            [x_start, y_max],
        ]
        ax.add_patch(
            patches.Polygon(
                path,
                linewidth=rc.linewidth_thin,
                facecolor=rc.color,
                edgecolor=rc.color,
            )
        )
        if proc == "world":
            ax.plot(
                [ts_send[i], ts_send[i]],
                [-0.25, 2.25],
                color=rc.color,
                linewidth=rc.linewidth_thin,
                linestyle="dotted",
            )
    x_text = 1.02 * np.minimum(np.min(ts_send), np.min(ts_recv))
    ax.text(
        x_text,
        0.5,
        "agent",
        horizontalalignment="right",
        verticalalignment="center",
        color=rc.color,
        fontsize=rc.fontsize_small,
    )
    ax.text(
        x_text,
        1.5,
        "world",
        horizontalalignment="right",
        verticalalignment="center",
        color=rc.color,
        fontsize=rc.fontsize_small,
    )


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("_db_name")
    args = parser.parse_args()
    report_timing(args._db_name)


if __name__ == "__main__":
    cli()
