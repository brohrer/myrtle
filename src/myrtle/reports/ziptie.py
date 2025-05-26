import os
import numpy as np
import matplotlib.pyplot as plt
from myrtle.config import log_directory
from myrtle.reports.report_config import (
    array_1D_report,
    array_2D_report,
)


def report():
    log_subdir = "ziptie"
    cable_activities = np.load(
        os.path.join(log_directory, log_subdir, "cable_activities.npy")
    )
    bundle_activities = np.load(
        os.path.join(log_directory, log_subdir, "bundle_activities.npy")
    )
    mapping = np.load(os.path.join(log_directory, log_subdir, "mapping.npy"))
    # n_cables_by_bundle = np.load(
    #     os.path.join(log_directory, log_subdir, "n_cables_by_bundle.npy")
    # )
    nucleation_energy = np.load(
        os.path.join(log_directory, log_subdir, "nucleation_energy.npy")
    )
    nucleation_mask = np.load(
        os.path.join(log_directory, log_subdir, "nucleation_mask.npy")
    )
    agglomeration_energy = np.load(
        os.path.join(log_directory, log_subdir, "agglomeration_energy.npy")
    )
    agglomeration_mask = np.load(
        os.path.join(log_directory, log_subdir, "agglomeration_mask.npy")
    )

    print()
    print("ziptie")

    try:
        n_bundles = np.max(np.where(mapping > -1)[0]) + 1
        print(f"  {n_bundles} feature bundles")
    except ValueError:
        print("  no feature bundles created yet")

    array_2D_report(
        nucleation_energy.transpose(),
        drop_zeros=True,
        name="nucleation energy",
        xlabel="sensors",
        ylabel="sensors",
    )
    plt.savefig(
        os.path.join(log_directory, log_subdir, "nucleation_energy.png"),
        dpi=300,
    )
    plt.show()

    array_2D_report(
        nucleation_mask.transpose(),
        drop_zeros=True,
        name="nucleation mask",
        xlabel="sensors",
        ylabel="sensors",
    )
    plt.savefig(
        os.path.join(log_directory, log_subdir, "nucleation_mask.png"),
        dpi=300,
    )
    plt.show()

    array_2D_report(
        agglomeration_energy.transpose(),
        drop_zeros=True,
        name="agglomeration energy",
        xlabel="sensors",
        ylabel="features",
    )
    plt.savefig(
        os.path.join(log_directory, log_subdir, "agglomeration_energy.png"),
        dpi=300,
    )
    plt.show()

    array_2D_report(
        agglomeration_mask.transpose(),
        drop_zeros=True,
        name="agglomeration mask",
        xlabel="sensors",
        ylabel="features",
    )
    plt.savefig(
        os.path.join(log_directory, log_subdir, "agglomeration_mask.png"),
        dpi=300,
    )
    plt.show()

    # print("mapping")
    # print(mapping)
    # print("n_cable_by_bundle")
    # print(n_cables_by_bundle)

    array_1D_report(cable_activities, "cable (sensor) activities")

    array_1D_report(bundle_activities, "bundle (feature) activities")


"""
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import dsmq.client
from myrtle.agents.tools import retrieve_ziptie_info
from myrtle.config import mq_host, mq_port
from myrtle.reports.report_config import *

history_length_short = 5
history_length_med = 20
history_length_long = 100
history_length_super_long = 1000

_connection_wait = 2.0  # seconds
_n_tries = 10
"""
"""
def report_ziptie():
    mq = dsmq.client.connect(mq_host, mq_port)
    time.sleep(_connection_wait)

    n_images = 3
    fig, axes_list = blank_images(n_images)
    # From bottom to top
    ax_agglomeration, ax_nucleation, ax_bundles = axes_list

    try:
        (
            n_sensors,
            n_bundles,
            bundle_mapping,
            nucleation_energy,
            agglomeration_energy,
        ) = retrieve_ziptie_info(mq)
    except TypeError:
        print("Try again in a few seconds.")
        return

    print()
    print("Ziptie")
    print(f"    {n_sensors} sensors")
    print(f"    {n_bundles} features")

    cmap = LinearSegmentedColormap.from_list("matrix", ["black", color])

    ax_bundles.imshow(
        bundle_mapping.transpose(),
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
    ax_bundles.set_xlabel("bundles", color=color, fontsize=fontsize_small)

    ax_nucleation.imshow(
        nucleation_energy.transpose(),
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
    ax_nucleation.set_xlabel("nucleation energy", color=color, fontsize=fontsize_small)

    ax_agglomeration.imshow(
        agglomeration_energy.transpose(),
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        interpolation="nearest",
    )
    ax_agglomeration.set_xlabel("agglomeration energy", color=color, fontsize=fontsize_small)

    plt.show()
"""

if __name__ == "__main__":
    report()
