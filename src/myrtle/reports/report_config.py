import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# light mode
# background_color = "white"
# color = "black"

# Matrix mode
background_color = "black"
color = "cadetblue"  # #5F9EA0, CSS color

linewidth_ultra_thick = 24
linewidth_thick = 2
linewidth_med = 1
linewidth_thin = 0.5
linewidth_super_thin = 0.3

fontsize_small = 8
fontsize_med = 12
fontsize_large = 16

figsize = (6, 9)

epsilon = 1e-8


def blank_report(n_axes, figsize=figsize):
    left_border = 0.18
    right_border = 0.13
    bottom_border = 0.38
    top_border = 0.18

    fig = plt.figure(figsize=figsize, facecolor=background_color)

    axes_list = []

    for i_axes in range(n_axes):
        left = left_border
        bottom = (i_axes + bottom_border) / n_axes
        width = 1 - (left_border + right_border)
        height = (1 - bottom_border - top_border) / n_axes
        ax = fig.add_axes((left, bottom, width, height))
        ax.set_facecolor(background_color)
        ax.tick_params(
            axis="both",
            color=color,
            labelcolor=color,
            labelsize=fontsize_small,
        )
        # ax.spines["top"].set_color(color)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color(color)
        # ax.spines["left"].set_color(color)
        ax.spines["left"].set_visible(False)
        # ax.spines["right"].set_color(color)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=color, linestyle=":", linewidth=linewidth_thin)
        axes_list.append(ax)

    return fig, axes_list


def blank_images(n_axes, figsize=figsize):
    left_border = 0.1
    right_border = 0.08
    bottom_border = 0.15
    top_border = 0.08

    fig = plt.figure(figsize=figsize, facecolor=background_color)

    axes_list = []

    for i_axes in range(n_axes):
        left = left_border
        bottom = (i_axes + bottom_border) / n_axes
        width = 1 - (left_border + right_border)
        height = (1 - bottom_border - top_border) / n_axes
        ax = fig.add_axes((left, bottom, width, height))
        ax.set_facecolor(background_color)
        ax.tick_params(
            axis="both",
            color=color,
            labelcolor=color,
            labelsize=fontsize_small,
        )

        ax.grid(False)
        ax.spines["top"].set_color(color)
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_color(color)

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(True)

        ax.spines["top"].set_linewidth(linewidth_thin)
        ax.spines["bottom"].set_linewidth(linewidth_thin)
        ax.spines["left"].set_linewidth(linewidth_thin)
        ax.spines["right"].set_linewidth(linewidth_thin)

        axes_list.append(ax)

    return fig, axes_list


def array_1D_report(arr, name=""):
    print()
    print(f"{name}, 1D array")
    print(f"  size {arr.size}")
    i_nonzero = np.where(arr > 0)[0]
    if i_nonzero.size < 20:
        print("  nonzero elements")
        for i in i_nonzero:
            print(f"    {i}  : {arr[i]}")


def array_2D_report(arr, drop_zeros=False, name="", xlabel="", ylabel=""):
    fig_width = 16
    fig_height = 9

    left_im = 0.1
    right_im = 0.6
    bottom_im = 0.15
    top_im = 0.9

    left_hist = 0.65
    right_hist = 0.92
    bottom_hist = 0.6
    top_hist = 0.9

    left_log = left_hist
    right_log = right_hist
    bottom_log = 0.15
    top_log = 0.45

    print()
    print(f"{name}, 2D array")
    print(f"  shape {arr.shape}")
    i_row_nonzero, i_col_nonzero = np.where(arr > 0)
    if i_row_nonzero.size < 20:
        print("  nonzero elements")
        for i_row, i_col in zip(i_row_nonzero, i_col_nonzero):
            print(f"    [{i_row}, {i_col}]  : {arr[i_row, i_col]}")

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=background_color)

    ax_im = fig.add_axes((left_im, bottom_im, right_im - left_im, top_im - bottom_im))
    ax_im.set_facecolor(background_color)
    ax_im.tick_params(
        axis="both",
        color=color,
        labelcolor=color,
        labelsize=fontsize_small,
    )

    ax_im.grid(False)
    ax_im.spines["top"].set_color(color)
    ax_im.spines["bottom"].set_color(color)
    ax_im.spines["left"].set_color(color)
    ax_im.spines["right"].set_color(color)

    ax_im.spines["top"].set_visible(True)
    ax_im.spines["bottom"].set_visible(True)
    ax_im.spines["left"].set_visible(True)
    ax_im.spines["right"].set_visible(True)

    ax_im.spines["top"].set_linewidth(linewidth_thin)
    ax_im.spines["bottom"].set_linewidth(linewidth_thin)
    ax_im.spines["left"].set_linewidth(linewidth_thin)
    ax_im.spines["right"].set_linewidth(linewidth_thin)

    cmap = LinearSegmentedColormap.from_list("matrix", ["black", color])

    ax_im.imshow(
        arr,
        cmap=cmap,
        interpolation="nearest",
    )
    ax_im.set_xlabel(xlabel, color=color, fontsize=fontsize_small)
    ax_im.set_ylabel(ylabel, color=color, fontsize=fontsize_small)

    ## Histogram of values
    ax_hist = fig.add_axes(
        (left_hist, bottom_hist, right_hist - left_hist, top_hist - bottom_hist)
    )
    ax_hist.set_facecolor(background_color)
    ax_hist.tick_params(
        axis="both",
        color=color,
        labelcolor=color,
        labelsize=fontsize_small,
    )
    # ax_hist.spines["top"].set_color(color)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["bottom"].set_color(color)
    # ax_hist.spines["left"].set_color(color)
    ax_hist.spines["left"].set_visible(False)
    # ax_hist.spines["right"].set_color(color)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.grid(axis="y", color=color, linestyle=":", linewidth=linewidth_thin)

    if drop_zeros:
        values = arr[np.where(np.abs(arr) > epsilon)].ravel()
    else:
        values = arr.ravel()

    plot_distribution(ax_hist, values)
    ax_hist.set_ylabel("occurrences", color=color, fontsize=fontsize_small)

    ## Log count histogram of values
    ax_log = fig.add_axes(
        (left_log, bottom_log, right_log - left_log, top_log - bottom_log)
    )
    ax_log.set_facecolor(background_color)
    ax_log.tick_params(
        axis="both",
        color=color,
        labelcolor=color,
        labelsize=fontsize_small,
    )
    # ax_log.spines["top"].set_color(color)
    ax_log.spines["top"].set_visible(False)
    ax_log.spines["bottom"].set_color(color)
    # ax_log.spines["left"].set_color(color)
    ax_log.spines["left"].set_visible(False)
    # ax_log.spines["right"].set_color(color)
    ax_log.spines["right"].set_visible(False)
    ax_log.grid(axis="y", color=color, linestyle=":", linewidth=linewidth_thin)

    plot_distribution(ax_log, values, log=True)
    ax_log.set_xlabel(name, color=color, fontsize=fontsize_med)
    ax_log.set_ylabel("log(occurrences)", color=color, fontsize=fontsize_small)

    return fig, ax_im, ax_hist, ax_log


def plot_distribution(ax, values, bins=100, title_text="", log=False):
    ax.hist(values, bins=bins, color=color, log=log)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    left_title(ax, title_text)


def bin_avg(x, y, width):
    """
    Returns the right edges of each bin.
    """
    x_max = np.max(x)
    y_binned = []
    x_bins = []
    i_bin = 0
    while True:
        i_bin += 1
        # If this is a partial bin, don't include it.
        # Partial bins tend to have high variability and generate more
        # confusion from freakishly high or low values, than they provide
        # understanding.
        if i_bin * width > x_max:
            break

        x_bins.append(i_bin * width)
        i_bin_x = np.where(np.logical_and(x >= (i_bin - 1) * width, x < i_bin * width))
        if i_bin_x[0].size > 0:
            y_binned.append(np.mean(y[i_bin_x]))
        else:
            y_binned.append(np.nan)

    return x_bins, y_binned


def left_title(ax, text):
    xmin, xmax = ax.get_xlim()
    x_text = xmin - (xmax - xmin) * 0.13
    ax.text(
        x_text,
        ax.get_ylim()[1],
        text,
        horizontalalignment="left",
        verticalalignment="bottom",
        color=color,
        fontsize=fontsize_small,
    )
