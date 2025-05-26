import os
import numpy as np
import matplotlib.pyplot as plt
from myrtle.config import log_directory
from myrtle.reports.report_config import (
    array_1D_report,
    array_2D_report,
)


def report():
    print("reporting")
    curiosities = np.load(os.path.join(log_directory, "fnc", "curiosities.npy"))
    features = np.load(os.path.join(log_directory, "fnc", "features.npy"))
    predicted_reward = np.load(
        os.path.join(log_directory, "fnc", "predicted_reward.npy")
    )
    predictions = np.load(os.path.join(log_directory, "fnc", "predictions.npy"))
    # previous_sensors = np.load(
    #     os.path.join(log_directory, "fnc", "previous_sensors.npy")
    # )
    sensors = np.load(os.path.join(log_directory, "fnc", "sensors.npy"))

    print()
    print("FNC")

    array_1D_report(sensors, "sensor activities")

    array_1D_report(features, "feature activities")

    array_1D_report(predicted_reward, "predicted reward, by action")

    array_2D_report(
        predictions,
        drop_zeros=True,
        name="predicted sensors",
        xlabel="features",
        ylabel="actions",
    )
    plt.savefig(os.path.join(log_directory, "fnc", "predicted_sensors.png"), dpi=300)
    plt.show()

    array_2D_report(
        curiosities.transpose(),
        drop_zeros=True,
        name="curiosities",
        xlabel="features",
        ylabel="actions",
    )
    plt.savefig(os.path.join(log_directory, "fnc", "curiosities.png"), dpi=300)
    plt.show()

    print()


if __name__ == "__main__":
    report()
