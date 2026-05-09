#!/usr/bin/env python3
"""Allan-variance analysis for the EKF sensor-noise model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from AI_UAV_Tests.ekf.ekf_diagnostics import (
    allan_deviation,
    estimate_allan_noise_parameters,
)
from AI_UAV_Tests.ekf.sensors_ekf import EKFSensorNoise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Allan-variance analysis on the EKF sensor-noise model."
    )
    parser.add_argument("--samples", type=int, default=200000, help="Number of stationary IMU samples")
    parser.add_argument("--dt", type=float, default=0.002, help="Sampling period in seconds")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory for plots and text report")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args()


def collect_stationary_imu_samples(n_samples: int, dt: float):
    sensor = EKFSensorNoise(
        sample_turn_on_bias_once=True,
        gyro_turn_on_bias_sigma=0.0,
    )
    pos = np.zeros(3, dtype=float)
    vel = np.zeros(3, dtype=float)
    rot = np.zeros(3, dtype=float)
    omega = np.zeros(3, dtype=float)
    acc = np.zeros(3, dtype=float)

    gyro = np.zeros((n_samples, 3), dtype=float)
    accel = np.zeros((n_samples, 3), dtype=float)
    for idx in range(n_samples):
        _, _, _, noisy_rate, noisy_acc = sensor.add_noise(pos, vel, rot, omega, acc, dt)
        gyro[idx] = noisy_rate
        accel[idx] = noisy_acc
    return sensor, gyro, accel


def summarise_sensor_model(sensor: EKFSensorNoise, gyro: np.ndarray, accel: np.ndarray, dt: float) -> str:
    lines = [
        "Allan variance IMU noise analysis",
        "note: this characterises the current simulated EKF sensor model, not real hardware logs.",
        f"dt={dt:.6f} s  samples={gyro.shape[0]}",
        "",
    ]

    for name, samples, unit in (
        ("gyro", gyro, "rad/s"),
        ("accel", accel, "m/s^2"),
    ):
        coeffs = []
        lines.append(f"{name} ({unit})")
        for axis in range(3):
            estimate = estimate_allan_noise_parameters(samples[:, axis], dt)
            white = float(estimate.get("white_noise_coeff", np.nan))
            rw = float(estimate.get("random_walk_coeff", np.nan))
            coeffs.append((white, rw))
            lines.append(
                f"  axis {axis}: white_noise_coeff={white:.6e}  random_walk_coeff={rw:.6e}"
            )
        mean_white = np.nanmean([c[0] for c in coeffs])
        mean_rw = np.nanmean([c[1] for c in coeffs])
        lines.append(
            f"  mean:  white_noise_coeff={mean_white:.6e}  random_walk_coeff={mean_rw:.6e}"
        )
        lines.append("")

    lines.append("Sensor model parameters")
    lines.append(f"  gyro_noise_density      = {sensor.gyro_noise_density:.6e}")
    lines.append(f"  gyro_random_walk       = {sensor.gyro_random_walk:.6e}")
    lines.append(f"  acc_static_noise_std   = {sensor.acc_static_noise_std:.6e}")
    lines.append(f"  acc_dynamic_noise_ratio= {sensor.acc_dynamic_noise_ratio:.6e}")
    return "\n".join(lines) + "\n"


def plot_allan_curves(gyro: np.ndarray, accel: np.ndarray, dt: float, save_dir: str | None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, samples, unit in (
        (axs[0], "Gyro", gyro, "rad/s"),
        (axs[1], "Accel", accel, "m/s²"),
    ):
        mean_curves = []
        ref_taus = None
        for axis in range(3):
            taus, adev = allan_deviation(samples[:, axis], dt)
            ref_taus = taus if ref_taus is None else ref_taus
            mean_curves.append(adev)
            ax.loglog(taus, adev, alpha=0.35, linewidth=1.0, label=f"axis {axis}")
        mean_curve = np.mean(np.vstack(mean_curves), axis=0)
        ax.loglog(ref_taus, mean_curve, color="black", linewidth=2.0, label="mean")
        ax.set_title(f"{name} Allan Deviation")
        ax.set_xlabel("Cluster time τ [s]")
        ax.set_ylabel(f"Allan deviation [{unit}]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "imu_allan_deviation.png", dpi=150, bbox_inches="tight")
    return fig


def main():
    args = parse_args()
    if args.show:
        matplotlib.use("TkAgg")

    sensor, gyro, accel = collect_stationary_imu_samples(args.samples, args.dt)
    report = summarise_sensor_model(sensor, gyro, accel, args.dt)
    print(report)

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.save_dir) / "imu_allan_report.txt").write_text(report, encoding="utf-8")

    fig = plot_allan_curves(gyro, accel, args.dt, args.save_dir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
