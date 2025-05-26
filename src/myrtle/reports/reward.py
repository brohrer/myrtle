import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sqlogging import logging
from myrtle.config import log_directory
import myrtle.reports.report_config as config


def report_reward(db_name):
    results = retrieve_reward(db_name)
    reward = results[:, 0]
    step = results[:, 1]
    episode = results[:, 2]
    current_episode = int(np.max(episode))
    step_ep = step[np.where(episode == current_episode)]
    reward_ep = reward[np.where(episode == current_episode)]

    n_plots = 3
    fig, axes_list = config.blank_report(n_plots)
    # From bottom to top
    ax_all, ax_med, ax_fast = axes_list

    max_history = 100
    ax_fast.plot(
        step_ep[-max_history:],
        reward_ep[-max_history:],
        color=config.color,
        linewidth=config.linewidth_thick,
    )

    med_bin_width = 100
    step_med, reward_med = config.bin_avg(step_ep, reward_ep, med_bin_width)
    ax_med.plot(
        step_med[-max_history:],
        reward_med[-max_history:],
        color=config.color,
        linewidth=config.linewidth_thick,
    )
    ax_med.set_ylabel("reward", color=config.color, fontsize=config.fontsize_med)

    n_all_bins = 30
    all_bin_width = int(np.ceil(np.max(step) / n_all_bins))
    for i_episode in range(current_episode + 1):
        step_ep = step[np.where(episode == i_episode)]
        reward_ep = reward[np.where(episode == i_episode)]
        step_all, reward_all = config.bin_avg(step_ep, reward_ep, all_bin_width)
        if current_episode == 0:
            linewidth = config.linewidth_thick
        elif current_episode < 5:
            linewidth = config.linewidth_med
        elif current_episode < 10:
            linewidth = config.linewidth_thin
        else:
            linewidth = config.linewidth_super_thin
        ax_all.plot(
            step_all,
            reward_all,
            color=config.color,
            linewidth=linewidth,
        )
    ax_all.set_xlabel("steps", color=config.color, fontsize=config.fontsize_med)

    report_filename = f"reward_{db_name}.png"
    plt.savefig(os.path.join(log_directory, report_filename))

    plt.show()


def retrieve_reward(db_name):
    logger = logging.open_logger(
        name=db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT reward,
            step,
            episode
        FROM {db_name}
        WHERE process = 'world'
    """
    )
    results = np.array(result)
    return results


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("_db_name")
    args = parser.parse_args()
    report_reward(args._db_name)


if __name__ == "__main__":
    cli()
