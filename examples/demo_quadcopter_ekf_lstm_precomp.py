#!/usr/bin/env python3
"""Demo: QuadcopterEKF + LSTM waypoint pre-compensation during GPS dropout."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter
from AI_UAV_Tests.ekf_lstm_precomp import (
    BUFFER_LEN,
    DEFAULT_MODEL_PATH,
    TAKEOFF_Z,
    NUM_HORIZONS,
    STANDARD_TRAJECTORIES,
    QuadcopterPositionDriftLSTM,
    build_standard_mission,
    build_phase_dropout_windows,
    build_lstm_feature,
    compute_lstm_state_correction,
    run_reference_velocity_dropout_step,
    thrust_to_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LSTM reference pre-compensation with the 12-state QuadcopterEKF."
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="random",
        choices=["random", *STANDARD_TRAJECTORIES],
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dropout-duration", type=float, default=2.0)
    parser.add_argument("--dropout-margin", type=float, default=0.75)
    parser.add_argument("--playback-slowdown", type=float, default=1.0)
    return parser.parse_args()


def draw_line(bc, start: np.ndarray, end: np.ndarray, color: tuple[float, float, float], width: float = 2.0) -> None:
    bc.addUserDebugLine(
        lineFromXYZ=np.asarray(start, dtype=float).tolist(),
        lineToXYZ=np.asarray(end, dtype=float).tolist(),
        lineColorRGB=list(color),
        lineWidth=width,
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model not found at {model_path}. "
            "Run collect_quadcopter_ekf_drift_data.py and train_quadcopter_ekf_drift_lstm.py first."
        )

    model = QuadcopterPositionDriftLSTM(horizon_steps=NUM_HORIZONS, hidden_size=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    env = DroneMissionEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human" if args.render else None,
    )
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP,
    )
    env.reset()

    trajectory_name = (
        args.trajectory
        if args.trajectory != "random"
        else str(np.random.choice(STANDARD_TRAJECTORIES))
    )
    mission = build_standard_mission(trajectory_name)

    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    ekf = PhoenixEKFAdapter(
        dt=env.TIME_STEP,
        use_attitude_measurements=True,
        use_velocity_measurements=True,
    )
    ekf.decouple_after_s = None
    ekf.reset(
        position=env.drone.xyz,
        velocity=env.drone.xyz_dot,
        attitude=env.drone.rpy,
        rates=env.drone.rpy_dot,
    )
    estimate = ekf.ekf.as_dict()

    imu_buffer = np.zeros((BUFFER_LEN, 7), dtype=np.float32)
    buffer_idx = 0
    last_lstm_preds = np.zeros((NUM_HORIZONS, 3), dtype=np.float32)
    steps_since_dropout = 0
    prev_gps_ok = True

    total_time = float(mission.total_time)
    steps = int(np.ceil(total_time / env.TIME_STEP)) + 1
    dropout_windows = build_phase_dropout_windows(
        mission,
        duration_s=args.dropout_duration,
        margin_s=args.dropout_margin,
        max_windows=2,
    )

    print(f"\n{'=' * 76}")
    print("QuadcopterEKF + LSTM Pre-Compensation Demo")
    print(f"{'=' * 76}")
    print(f"Trajectory          : {trajectory_name}")
    print(f"Mission time        : {total_time:.1f}s")
    print(f"Dropout windows     : {', '.join(f'{s:.1f}-{e:.1f}s' for s, e in dropout_windows) or 'none'}")
    print(f"Model               : {model_path}")
    print(f"{'=' * 76}\n")

    log_t = []
    log_true = []
    log_est = []
    log_ref = []
    log_nom_ref = []
    log_dropout = []
    log_error = []
    log_shift = []
    prev_draw_pos = np.asarray(env.drone.xyz, dtype=float).copy()

    try:
        for k in range(steps):
            if args.render:
                time.sleep(env.TIME_STEP * max(args.playback_slowdown, 0.0))

            t = float(k * env.TIME_STEP)
            gps_ok = not any(start <= t < end for start, end in dropout_windows)

            true_pos = np.asarray(env.drone.xyz, dtype=float)
            vel_true = np.asarray(env.drone.xyz_dot, dtype=float)
            ang_true = np.asarray(env.drone.rpy, dtype=float)
            rate_true = np.asarray(env.drone.rpy_dot, dtype=float)
            nominal_ref, nominal_vel = mission(t)

            phase_name = mission.phase_name_at(t)
            ref_target = nominal_ref.copy()
            ref_vel = nominal_vel.copy()
            if t < 3.0:
                ref_target = np.array([0.0, 0.0, TAKEOFF_Z], dtype=float)
                ref_vel = np.zeros(3, dtype=float)
            z_ref = float(ref_target[2])

            if gps_ok:
                pos_ctrl = true_pos.copy()
                vel_ctrl = vel_true.copy()
            else:
                pos_ctrl = np.array([estimate["x"][0], estimate["x"][1], true_pos[2]], dtype=float)
                vel_ctrl = vel_true.copy()

            quad.inject_external_state(pos_ctrl, vel_ctrl, ang_true, rate_true)
            ctrl = quad.step(ref_target, ref_vel, z_ref=z_ref)

            action = np.zeros(4, dtype=np.float32)
            action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
            action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

            _obs, _reward, done, truncated, _info = env.step(action)

            if gps_ok:
                estimate = ekf.step(
                    motor_omega=np.asarray(ctrl["omega_cmd"], dtype=float),
                    position=env.drone.xyz,
                    velocity=env.drone.xyz_dot,
                    attitude=env.drone.rpy,
                    rates=env.drone.rpy_dot,
                    dropout_active=False,
                    dt=env.TIME_STEP,
                )
            else:
                estimate = run_reference_velocity_dropout_step(
                    ekf,
                    motor_forces=np.asarray(env.drone.y, dtype=float),
                    position=env.drone.xyz,
                    velocity=env.drone.xyz_dot,
                    attitude=env.drone.rpy,
                    rates=env.drone.rpy_dot,
                    velocity_ref=nominal_vel,
                    position_ref=nominal_ref,
                    dt=env.TIME_STEP,
                )

            if prev_gps_ok and not gps_ok:
                steps_since_dropout = 0
            elif not prev_gps_ok and gps_ok:
                steps_since_dropout = 0
            elif not gps_ok:
                steps_since_dropout += 1
            t_norm = min(steps_since_dropout / 2500.0, 1.0)

            feature = build_lstm_feature(
                np.asarray(estimate["v"], dtype=float),
                np.asarray(estimate["measurement"]["rates"], dtype=float),
                t_norm,
            )
            imu_buffer[buffer_idx % BUFFER_LEN] = feature
            buffer_idx += 1

            if buffer_idx >= BUFFER_LEN and not gps_ok:
                ordered = np.roll(imu_buffer, -(buffer_idx % BUFFER_LEN), axis=0)
                tensor = torch.from_numpy(ordered.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = model(tensor)
                last_lstm_preds = preds[0].cpu().numpy()
            elif gps_ok:
                last_lstm_preds[:] = 0.0

            prev_gps_ok = gps_ok

            metric_correction = np.zeros(3, dtype=float)
            if not gps_ok and buffer_idx >= BUFFER_LEN and phase_name in {"circle", "square", "helix", "sine", "figure_eight"}:
                metric_correction = compute_lstm_state_correction(
                    lstm_preds=last_lstm_preds,
                    dt=env.TIME_STEP,
                    ref_vel=nominal_vel,
                    dropout_time_s=float(ekf.dropout_time),
                )
            corrected_estimate_metric = np.asarray(estimate["x"], dtype=float).copy() - metric_correction

            ekf_error = float(np.linalg.norm(corrected_estimate_metric - np.asarray(env.drone.xyz, dtype=float)))
            shift_mag = float(np.linalg.norm(metric_correction))

            if args.render and k > 0:
                draw_line(
                    env.bc,
                    prev_draw_pos,
                    np.asarray(env.drone.xyz, dtype=float),
                    (0.95, 0.15, 0.15) if not gps_ok else (0.10, 0.35, 0.95),
                )
            prev_draw_pos = np.asarray(env.drone.xyz, dtype=float).copy()

            log_t.append(t)
            log_true.append(np.asarray(env.drone.xyz, dtype=float))
            log_est.append(corrected_estimate_metric)
            log_ref.append(np.asarray(ref_target, dtype=float))
            log_nom_ref.append(np.asarray(nominal_ref, dtype=float))
            log_dropout.append(not gps_ok)
            log_error.append(ekf_error)
            log_shift.append(shift_mag)

            if k % 40 == 0:
                print(
                    f"t={t:6.2f}s  gps={'ON ' if gps_ok else 'OFF'}  "
                    f"ekf_err={ekf_error:5.3f}m  shift={shift_mag:5.3f}m  "
                    f"phase={phase_name:>12}"
                )

            if done or truncated or t >= total_time:
                break
    finally:
        env.close()

    if not log_t:
        return

    t_arr = np.asarray(log_t, dtype=float)
    true_arr = np.asarray(log_true, dtype=float)
    est_arr = np.asarray(log_est, dtype=float)
    ref_arr = np.asarray(log_ref, dtype=float)
    nom_ref_arr = np.asarray(log_nom_ref, dtype=float)
    dropout_mask = np.asarray(log_dropout, dtype=bool)
    err_arr = np.asarray(log_error, dtype=float)
    shift_arr = np.asarray(log_shift, dtype=float)

    print(f"\nMean EKF error       : {np.mean(err_arr):.3f} m")
    if np.any(dropout_mask):
        print(f"Mean dropout error   : {np.mean(err_arr[dropout_mask]):.3f} m")
        print(f"Max dropout error    : {np.max(err_arr[dropout_mask]):.3f} m")
        print(f"Mean correction mag  : {np.mean(shift_arr[dropout_mask]):.3f} m")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    axs[0].plot(true_arr[:, 0], true_arr[:, 1], label="true path", linewidth=1.8)
    axs[0].plot(est_arr[:, 0], est_arr[:, 1], label="EKF estimate", linewidth=1.4)
    axs[0].plot(nom_ref_arr[:, 0], nom_ref_arr[:, 1], "--", label="nominal reference", linewidth=1.0)
    axs[0].plot(ref_arr[:, 0], ref_arr[:, 1], label="corrected reference", linewidth=1.0)
    axs[0].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")
    axs[0].set_title("XY Path")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].plot(t_arr, err_arr, label="corrected position error", linewidth=1.5)
    axs[1].plot(t_arr, shift_arr, label="applied correction magnitude", linewidth=1.2)
    for start, end in dropout_windows:
        axs[1].axvspan(start, end, color="0.9", alpha=0.5, zorder=0)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Metres")
    axs[1].set_title("Error and LSTM Correction Magnitude")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
