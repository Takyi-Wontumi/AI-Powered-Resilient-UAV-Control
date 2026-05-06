"""
Follow-path mission with dropout handling, driven by EKF state estimates.

Keys:
    H -> hover at current position
    R -> return to home
    L -> land vertically
    C -> clear dropout and resume trajectory
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.followpath_dropout_mission import (
    DroneFollowPathDropoutMissionEnv,
)
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter
from AI_UAV_Tests.sensors_ekf import EKFSensorNoise
from AI_UAV_Tests.trajectories_library import FlightMission

STATE_DIM = 12
AUTO_DROPOUT_WINDOWS_S = (
    (5.0, 2.5),
    (15.0, 2.5),
)
TRAIL_BLUE = [0.10, 0.35, 0.95]
TRAIL_RED = [0.95, 0.15, 0.15]
RECOVERY_POS_ERR_THRESHOLD_M = 0.15
def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


def reset_pid_integrators(quad: QuadcopterPID) -> None:
    quad.z_int = 0.0
    quad.att_int[:] = 0.0
    quad.rate_int[:] = 0.0


def key_was_triggered(keys: dict, key: str, bullet_client) -> bool:
    return any(
        ord(code) in keys and keys[ord(code)] & bullet_client.KEY_WAS_TRIGGERED
        for code in {key.lower(), key.upper()}
    )


def compute_group_nees(
    x_est: np.ndarray,
    x_true: np.ndarray,
    P: np.ndarray,
    indices: slice | np.ndarray,
) -> float:
    err = np.asarray(x_est, dtype=float)[indices] - np.asarray(x_true, dtype=float)[indices]
    P_sub = np.asarray(P, dtype=float)[indices, :][:, indices]
    return float(err @ np.linalg.solve(P_sub + 1.0e-12 * np.eye(err.size), err))


def compute_nis(innovation: np.ndarray, S: np.ndarray) -> float:
    innovation = np.asarray(innovation, dtype=float)
    return float(innovation @ np.linalg.solve(S, innovation))


def format_dropout_schedule() -> str:
    return ", ".join(
        f"{start:.1f}s to {start + duration:.1f}s"
        for start, duration in AUTO_DROPOUT_WINDOWS_S
    )


def mission_plan_lines() -> list[str]:
    return [
        "Mission plan:",
        "  1. Takeoff: 0.0s to 3.0s -> climb to 1.0 m",
        "  2. Hover: 3.0s to 21.0s -> hold station so both dropout windows occur in steady hover",
        "  3. Landing: 21.0s to 27.0s -> descend to ground",
    ]


def update_auto_dropout(
    env,
    t: float,
    next_window_idx: int,
    auto_active: bool,
) -> tuple[int, bool]:
    """Run the configured scheduled hover dropouts for visualization/testing."""
    if next_window_idx >= len(AUTO_DROPOUT_WINDOWS_S):
        return next_window_idx, auto_active

    start_s, duration_s = AUTO_DROPOUT_WINDOWS_S[next_window_idx]
    end_s = start_s + duration_s
    if not auto_active and start_s <= t < end_s:
        env.dropout_mgr.mode = "HOV"
        env.trigger_dropout()
        return next_window_idx, True
    if auto_active and t >= end_s:
        env.clear_dropout()
        return next_window_idx + 1, False
    return next_window_idx, auto_active


def add_trail_segment(env, start: np.ndarray, end: np.ndarray, dropout_active: bool) -> None:
    color = TRAIL_RED if dropout_active else TRAIL_BLUE
    env.bc.addUserDebugLine(
        lineFromXYZ=np.asarray(start, dtype=float).tolist(),
        lineToXYZ=np.asarray(end, dtype=float).tolist(),
        lineColorRGB=color,
        lineWidth=2.0,
        lifeTime=0.0,
    )


def motor_omega_from_applied_forces(env, ekf: PhoenixEKFAdapter) -> np.ndarray:
    motor_forces = np.asarray(env.drone.y, dtype=float).reshape(4)
    thrust_coeff = float(ekf.ekf.params.b)
    return np.sqrt(np.clip(motor_forces, 0.0, np.inf) / thrust_coeff)


def build_mission():
    mission = FlightMission(default_z=1.0, ground_z=0.0)
    mission.add_takeoff(duration=3.0, target_z=1.0)
    # Keep the auto-dropout evaluation inside a stationary hover segment so
    # both windows are guaranteed to be reached and recovery can be measured.
    mission.add_hover(duration=18.0, z=1.0)
    mission.add_landing(duration=6.0, ground_z=0.055)
    return mission


def hover_omega_from_ekf_params(ekf: PhoenixEKFAdapter) -> np.ndarray:
    params = ekf.ekf.params
    return np.full(4, float(np.sqrt(params.m * params.g / (4.0 * params.b))), dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the EKF-backed follow-path dropout mission example."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the PyBullet GUI. By default the example runs headless.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the end-of-run XYZ reference vs measured plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mission = build_mission()
    env = DroneFollowPathDropoutMissionEnv(
        trajectory_fn=mission,
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human" if args.render else None,
        observation_noise=1.0,
    )
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP,
    )
    obs, info = env.reset()

    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    ekf = PhoenixEKFAdapter(
        dt=env.TIME_STEP,
        use_attitude_measurements=True,
    )
    # Example-level dropout tuning: reduce covariance inflation and disable
    # post-dropout cross-covariance decoupling for this hover-window test.
    ekf.decouple_after_s = None
    ekf.reset(
        position=env.drone.xyz,
        velocity=env.drone.xyz_dot,
        attitude=env.drone.rpy,
        rates=env.drone.rpy_dot,
    )
    estimate = ekf.ekf.as_dict()
    omega_hover = hover_omega_from_ekf_params(ekf)

    print("Mission started with EKF state feedback.")
    if args.render:
        print("Keys: H=hover, R=return-home, L=land, C=clear dropout")
        print(f"Automatic hover dropout windows: {format_dropout_schedule()}")
    else:
        print("Running headless. Use --render to enable the PyBullet GUI and keyboard dropout controls.")
    for line in mission_plan_lines():
        print(line)

    last_ctrl_pos = env.drone.xyz.copy()
    noise_gen     = EKFSensorNoise(
        sample_turn_on_bias_once=True,
        gyro_turn_on_bias_sigma=0.0,
    )

    steps = int(mission.total_time / env.TIME_STEP)
    log_t = []
    log_ref = []
    log_estimate = []
    log_measurement = []
    log_true = []
    log_nees = []
    log_nees_trans = []
    log_nees_att_rate = []
    log_nis_norm = []
    log_dropout_active = []
    log_measurement_dim = []
    prev_dropout_active = False
    hold_ctrl_pos = np.asarray(env.drone.xyz, dtype=float).copy()
    trail_prev_pos = np.asarray(env.drone.xyz, dtype=float).copy()
    auto_dropout_next_idx = 0
    auto_dropout_active = False
    for k in range(steps):
        if args.render:
            time.sleep(env.TIME_STEP)
        wall_time = float(k * env.TIME_STEP)

        auto_dropout_next_idx, auto_dropout_active = update_auto_dropout(
            env,
            wall_time,
            auto_dropout_next_idx,
            auto_dropout_active,
        )

        if args.render:
            keys = env.bc.getKeyboardEvents()
            if key_was_triggered(keys, "h", env.bc):
                env.dropout_mgr.mode = "HOV"
                env.trigger_dropout()
            if key_was_triggered(keys, "r", env.bc):
                env.dropout_mgr.mode = "RTH"
                env.trigger_dropout()
            if key_was_triggered(keys, "l", env.bc):
                env.dropout_mgr.mode = "LAND"
                env.trigger_dropout()
            if key_was_triggered(keys, "c", env.bc):
                env.clear_dropout()

        dropout_active = bool(env.dropout_mgr.active)
        dropout_mode = env.dropout_mgr.mode if dropout_active else "NONE"
        if dropout_active and not prev_dropout_active:
            hold_ctrl_pos = np.asarray(env.get_mission_reference(), dtype=float).copy()
            last_ctrl_pos = hold_ctrl_pos.copy()
            reset_pid_integrators(quad)
        elif not dropout_active and prev_dropout_active:
            reset_pid_integrators(quad)

        if dropout_active:
            pos_ref = np.asarray(env.get_mission_reference(), dtype=float)
            vel_ref = np.zeros(3, dtype=float)
        else:
            pos_ref, vel_ref = env.current_reference()
            pos_ref = np.asarray(pos_ref, dtype=float)
            vel_ref = np.asarray(vel_ref, dtype=float)

        ekf_dropout_active = False
        if dropout_active and dropout_mode == "HOV":
            n_pos, n_vel, n_att, n_rate, _ = noise_gen.add_noise(
                env.drone.xyz,
                env.drone.xyz_dot,
                env.drone.rpy,
                env.drone.rpy_dot,
                np.zeros(3, dtype=float),
                env.TIME_STEP,
            )
            ctrl_pos = hold_ctrl_pos.copy()
            ctrl_vel = np.zeros(3, dtype=float)
            # Keep altitude feedback alive so hover can bleed off climb/descent
            # momentum without re-enabling XY drift from the dropout estimator.
            ctrl_pos[2] = float(n_pos[2])
            ctrl_vel[2] = float(n_vel[2])
            quad.inject_external_state(
                ctrl_pos,
                ctrl_vel,
                n_att,
                n_rate,
            )
            last_ctrl_pos = ctrl_pos.copy()
            ekf_dropout_active = True
        elif dropout_active and dropout_mode in {"RTH", "LAND"}:
            n_pos, n_vel, n_att, n_rate, _ = noise_gen.add_noise(
                env.drone.xyz, env.drone.xyz_dot, env.drone.rpy, env.drone.rpy_dot,
                np.zeros(3, dtype=float), env.TIME_STEP,
            )
            quad.inject_external_state(n_pos, n_vel, n_att, n_rate)
            last_ctrl_pos = np.asarray(n_pos).copy()
        else:
            quad.inject_external_state(
                estimate["x"],
                estimate["v"],
                estimate["ang"],
                estimate["rate"],
            )
            last_ctrl_pos = np.asarray(estimate["x"]).copy()

        z_ref = float(pos_ref[2])
        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)

        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)
        auto_hover_pause_active = (
            auto_dropout_active
            and dropout_active
            and dropout_mode == "HOV"
        )
        if auto_hover_pause_active:
            # Keep the mission reference continuous after the scheduled hover
            # dropout by pausing trajectory time while wall-clock time advances.
            env.mission_time = max(0.0, float(env.mission_time - env.TIME_STEP))
        if args.render:
            add_trail_segment(
                env,
                trail_prev_pos,
                np.asarray(env.drone.xyz, dtype=float),
                dropout_active=dropout_active,
            )
        trail_prev_pos = np.asarray(env.drone.xyz, dtype=float).copy()
        measurement, H, R, noisy_state = ekf.build_noisy_measurement(
            position=env.drone.xyz,
            velocity=env.drone.xyz_dot,
            attitude=env.drone.rpy,
            rates=env.drone.rpy_dot,
            dropout_active=ekf_dropout_active,
            dt=env.TIME_STEP,
        )
        if ekf_dropout_active:
            ekf.dropout_time += env.TIME_STEP
            ekf.ekf.predict_dropout(
                omega=omega_hover,
                dt=env.TIME_STEP,
                dropout_time=ekf.dropout_time,
            )
            if (
                ekf.decouple_after_s is not None
                and ekf.dropout_time > float(ekf.decouple_after_s)
            ):
                ekf.ekf.decouple_all_groups()
        else:
            if ekf.dropout_time != 0.0:
                ekf.dropout_time = 0.0
            ekf.ekf.predict(omega=ctrl["omega_cmd"], dt=env.TIME_STEP)

        innovation = ekf.ekf.innovation(measurement, H)
        S = H @ ekf.ekf.P @ H.T + R
        nis_norm = compute_nis(innovation, S) / float(measurement.size)
        ekf.ekf.update(measurement=measurement, H=H, measurement_noise=R)

        estimate = ekf.ekf.as_dict()
        estimate["measurement"] = noisy_state
        x_true = np.concatenate(
            [
                np.asarray(env.drone.xyz, dtype=float),
                np.asarray(env.drone.xyz_dot, dtype=float),
                np.asarray(env.drone.rpy, dtype=float),
                np.asarray(env.drone.rpy_dot, dtype=float),
            ]
        )
        nees = ekf.ekf.compute_nees(x_true)
        nees_trans = compute_group_nees(
            estimate["state"],
            x_true,
            estimate["covariance"],
            slice(0, 6),
        )
        nees_att_rate = compute_group_nees(
            estimate["state"],
            x_true,
            estimate["covariance"],
            slice(6, 12),
        )
        prev_dropout_active = dropout_active
        estimated_pos = np.asarray(estimate["x"], dtype=float)
        measured_pos = np.asarray(estimate["measurement"]["position"], dtype=float)
        true_pos = np.asarray(env.drone.xyz, dtype=float)
        sim_time = float((k + 1) * env.TIME_STEP)
        log_t.append(sim_time)
        log_ref.append(pos_ref.copy())
        log_estimate.append(estimated_pos.copy())
        log_measurement.append(measured_pos.copy())
        log_true.append(true_pos.copy())
        log_nees.append(nees)
        log_nees_trans.append(nees_trans)
        log_nees_att_rate.append(nees_att_rate)
        log_nis_norm.append(nis_norm)
        log_dropout_active.append(dropout_active)
        log_measurement_dim.append(int(measurement.size))

        if k % 20 == 0:
            pos = estimate["x"]
            print(
                f"t={sim_time:6.2f}s  ref=({pos_ref[0]: .2f},{pos_ref[1]: .2f},{pos_ref[2]: .2f})  "
                f"est=({pos[0]: .2f},{pos[1]: .2f},{pos[2]: .2f})  "
                f"dropout={'ON' if dropout_active else 'OFF'}"
            )

        if done or truncated:
            print("Mission terminated.")
            break

    env.close()

    if log_t:
        t_arr = np.asarray(log_t, dtype=float)
        ref_arr = np.asarray(log_ref, dtype=float)
        estimate_arr = np.asarray(log_estimate, dtype=float)
        measurement_arr = np.asarray(log_measurement, dtype=float)
        true_arr = np.asarray(log_true, dtype=float)
        nees_arr = np.asarray(log_nees, dtype=float)
        nees_trans_arr = np.asarray(log_nees_trans, dtype=float)
        nees_att_rate_arr = np.asarray(log_nees_att_rate, dtype=float)
        nis_arr = np.asarray(log_nis_norm, dtype=float)
        dropout_mask = np.asarray(log_dropout_active, dtype=bool)
        nominal_mask = ~dropout_mask
        measurement_dim_arr = np.asarray(log_measurement_dim, dtype=int)
        pos_err_vec = estimate_arr - true_arr
        pos_err_norm = np.linalg.norm(pos_err_vec, axis=1)
        pos_err_ref_norm = np.linalg.norm(estimate_arr - ref_arr, axis=1)

        if not args.no_plot:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axis_labels = ["X [m]", "Y [m]", "Z [m]"]
            for i in range(3):
                for start_s, duration_s in AUTO_DROPOUT_WINDOWS_S:
                    axs[i].axvspan(
                        start_s,
                        start_s + duration_s,
                        color="0.9",
                        alpha=0.5,
                        zorder=0,
                    )
                nominal_series = np.where(nominal_mask, estimate_arr[:, i], np.nan)
                dropout_series = np.where(dropout_mask, estimate_arr[:, i], np.nan)
                axs[i].plot(t_arr, ref_arr[:, i], "--", color="black", linewidth=1.8, label="reference")
                axs[i].plot(
                    t_arr,
                    measurement_arr[:, i],
                    color="0.7",
                    alpha=0.9,
                    linewidth=1.0,
                    label="noisy measurement",
                )
                axs[i].plot(t_arr, nominal_series, color=TRAIL_BLUE, linewidth=1.8, label="EKF estimate (nominal)")
                axs[i].plot(t_arr, dropout_series, color=TRAIL_RED, linewidth=1.8, label="EKF estimate (dropout)")
                axs[i].set_ylabel(axis_labels[i])
                axs[i].grid(True, alpha=0.3)
                axs[i].legend(loc="best")

            axs[-1].set_xlabel("Time [s]")
            fig.suptitle("Reference, EKF Estimate, and Noisy Measurement")
            fig.tight_layout()

            fig_consistency, axs_consistency = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for ax in axs_consistency:
                for start_s, duration_s in AUTO_DROPOUT_WINDOWS_S:
                    ax.axvspan(
                        start_s,
                        start_s + duration_s,
                        color="0.9",
                        alpha=0.5,
                        zorder=0,
                    )
            axs_consistency[0].plot(t_arr, nees_arr, linewidth=1.5, label="NEES")
            axs_consistency[0].axhline(STATE_DIM, color="black", linestyle="--", linewidth=1.0, label="expected = 12")
            axs_consistency[0].set_ylabel("NEES")
            axs_consistency[0].grid(True, alpha=0.3)
            axs_consistency[0].legend(loc="best")

            axs_consistency[1].plot(t_arr, nis_arr, linewidth=1.5, label="NIS / DOF")
            axs_consistency[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="expected = 1.0")
            axs_consistency[1].set_xlabel("Time [s]")
            axs_consistency[1].set_ylabel("NIS / DOF")
            axs_consistency[1].grid(True, alpha=0.3)
            axs_consistency[1].legend(loc="best")
            fig_consistency.suptitle("EKF Consistency Metrics")
            fig_consistency.tight_layout()

        def print_consistency_block(label: str, mask: np.ndarray) -> None:
            sample_count = int(np.count_nonzero(mask))
            duration_s = sample_count * env.TIME_STEP
            if sample_count == 0:
                print(f"{label}: no samples")
                return
            mean_nees = float(np.mean(nees_arr[mask]))
            mean_nis = float(np.mean(nis_arr[mask]))
            mean_meas_dim = float(np.mean(measurement_dim_arr[mask]))
            print(
                f"{label}: duration={duration_s:.2f}s  samples={sample_count}  "
                f"mean NEES={mean_nees:.2f}  mean NIS/DOF={mean_nis:.3f}  "
                f"mean measurement dim={mean_meas_dim:.1f}"
            )

        def print_position_block(label: str, mask: np.ndarray) -> None:
            sample_count = int(np.count_nonzero(mask))
            if sample_count == 0:
                print(f"{label}: no samples")
                return
            rmse_true = float(np.sqrt(np.mean(pos_err_norm[mask] ** 2)))
            rmse_ref = float(np.sqrt(np.mean(pos_err_ref_norm[mask] ** 2)))
            max_err_true = float(np.max(pos_err_norm[mask]))
            print(
                f"{label}: position RMSE vs truth={rmse_true:.3f} m  "
                f"position RMSE vs ref={rmse_ref:.3f} m  "
                f"max position error vs truth={max_err_true:.3f} m"
            )

        def print_group_nees_block(label: str, mask: np.ndarray) -> None:
            sample_count = int(np.count_nonzero(mask))
            if sample_count == 0:
                print(f"{label}: no samples")
                return
            print(
                f"{label}: translational NEES={np.mean(nees_trans_arr[mask]):.2f} "
                f"(expected 6), attitude/rate NEES={np.mean(nees_att_rate_arr[mask]):.2f} "
                f"(expected 6)"
            )

        def print_recovery_block() -> None:
            if not AUTO_DROPOUT_WINDOWS_S:
                return
            for idx, (start_s, duration_s) in enumerate(AUTO_DROPOUT_WINDOWS_S, start=1):
                end_s = start_s + duration_s
                after_mask = t_arr >= end_s
                if not np.any(after_mask):
                    print(f"Recovery after dropout {idx}: no post-dropout samples")
                    continue
                after_indices = np.flatnonzero(after_mask)
                recovered_rel = np.flatnonzero(
                    pos_err_norm[after_indices] <= RECOVERY_POS_ERR_THRESHOLD_M
                )
                if recovered_rel.size == 0:
                    print(
                        f"Recovery after dropout {idx}: not recovered to "
                        f"{RECOVERY_POS_ERR_THRESHOLD_M:.2f} m position error threshold"
                    )
                    continue
                recovered_idx = after_indices[int(recovered_rel[0])]
                recovery_time = float(t_arr[recovered_idx] - end_s)
                print(
                    f"Recovery after dropout {idx}: {recovery_time:.2f} s "
                    f"to <= {RECOVERY_POS_ERR_THRESHOLD_M:.2f} m position error"
                )

        print(
            f"Mean consistency metrics: NEES={nees_arr.mean():.2f}, "
            f"NIS/DOF={nis_arr.mean():.3f}"
        )
        print_consistency_block("Nominal only", nominal_mask)
        print_consistency_block("Dropout only", dropout_mask)
        print_group_nees_block("Nominal grouped NEES", nominal_mask)
        print_group_nees_block("Dropout grouped NEES", dropout_mask)
        print_position_block("Nominal position errors", nominal_mask)
        print_position_block("Dropout position errors", dropout_mask)
        print_recovery_block()
        if not args.no_plot:
            plt.show()


if __name__ == "__main__":
    main()
