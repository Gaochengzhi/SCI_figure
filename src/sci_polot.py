import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import describe


def plot_frequency_analysis(
    df,
    speed_col="speed",
    accel_col="acc",
    output_file="frequency_analysis.png",
    speed_lower_limit=0.05,
    speed_upper_limit=99.95,
    accel_lower_limit=0.05,
    accel_upper_limit=99.95,
):
    speed_array = df[speed_col].to_numpy()
    accel_array = df[accel_col].to_numpy()
    fft_speed = (
        2.0 / len(speed_array) * np.abs(fft(speed_array)[: len(speed_array) // 2])
    )
    fft_accel = (
        2.0 / len(accel_array) * np.abs(fft(accel_array)[: len(accel_array) // 2])
    )
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ymin_speed = np.percentile(fft_speed, speed_lower_limit)
    ymax_speed = np.percentile(fft_speed, speed_upper_limit)
    ymin_accel = np.percentile(fft_accel, accel_lower_limit)
    ymax_accel = np.percentile(fft_accel, accel_upper_limit)

    ax1.plot(fft_speed, "-r", alpha=0.9)
    ax1.set_ylim(ymin_speed, ymax_speed)
    ax2.plot(fft_accel, "-b", alpha=0.3)
    ax2.set_ylim(ymin_accel, ymax_accel)

    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Amplitude (Speed)", color="r")
    ax2.set_ylabel("Amplitude (Accel)", color="b")

    mean_speed = np.mean(fft_speed)
    std_speed = np.std(fft_speed)
    mean_accel = np.mean(fft_accel)
    std_accel = np.std(fft_accel)

    ax1.text(
        0.05,
        0.90,
        f"Speed Mean Amplitude: {mean_speed:.2f}\nSpeed Amplitude Std Dev: {std_speed:.2f}",
        transform=ax1.transAxes,
        color="r",
    )

    ax1.text(
        0.05,
        0.70,
        f"Accel Mean Amplitude: {mean_accel:.2f}\nAccel Amplitude Std Dev: {std_accel:.2f}",
        transform=ax1.transAxes,
        color="b",
    )

    plt.tight_layout()
    plt.savefig(output_file)


def single_trajectories(
    df,
    id_col,
    row_num,
    col_num,
    col_titles=[
        "Vehicle Velocity",
        "Vehicle Acceleration",
        "Vehicle Angular Velocity Z",
        "Vehicle Control Steer",
    ],
    col_names=["Speed", "ACC", "angular_velocity_z", "control_steer"],
):
    vehicle_ids = df[id_col].unique()
    vehicle_subset = np.random.choice(vehicle_ids, row_num, replace=False)
    df = df[df[id_col].isin(vehicle_subset)]
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 48))
    # Loop through vehicles
    for i, vid in enumerate(vehicle_subset):
        df_vid = df[df[id_col] == vid]
        for j, col in enumerate(col_names):
            axs[i, j].plot(df_vid["time"], df_vid[col])
            axs[i, j].set_xlabel("Time")
            axs[i, j].set_ylabel(col_titles[j])
            if i == 0:
                axs[i, j].set_title(col_titles[j])
    fig.savefig(f"vehicle_plots.png")
    return df_vid
