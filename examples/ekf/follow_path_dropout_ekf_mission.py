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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.followpath_dropout_mission import (
    DroneFollowPathDropoutMissionEnv,
)
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.ekf.quadcopter_ekf import PhoenixEKFAdapter, QuadcopterEKF
from AI_UAV_Tests.ekf.sensors_ekf import EKFSensorNoise
from AI_UAV_Tests.core.trajectories_library import FlightMission

# Scheduled GPS-only dropout windows keyed by phase name and occurrence.
# Split the dropout across the trajectory: one early in the square and one
# near the end, instead of one continuous window.
AUTO_DROPOUT_PHASE_WINDOWS = (
    ("square", 0, 1.0, 2.5),
    ("square", 0, 8.5, 2.5),
)
TRAIL_BLUE = [0.10, 0.35, 0.95]
TRAIL_RED = [0.95, 0.15, 0.15]
RECOVERY_POS_ERR_THRESHOLD_M = 0.15
PSEUDO_VEL_STD_XY = 2.10
PSEUDO_VEL_STD_Z = 4.00
EKF_SIGMA_GYRO_BIAS = 0.001
EKF_Q_BIAS_FLOOR = 1.0e-8
EKF_INITIAL_GYRO_BIAS_STD = 0.001
GPS_RECOVERY_GATE_THRESHOLD = 11.34
GPS_RECOVERY_GATE_DURATION_S = 0.5
RECOVERY_BLEND_DURATION_S = 1.5
ATTITUDE_R_SCALE = 1.0
GYRO_R_SCALE = 1.0
VELOCITY_R_SCALE = 1.0
GPS_R_SCALE = 1.0
BARO_R_SCALE = 1.0
ODOM_R_SCALE = 1.0
FLOW_R_SCALE = 1.0
REAL_SENSOR_NAMES = ("attitude", "gyro", "velocity", "baro", "gps", "odom", "flow")
NAV_SENSOR_NAMES = {"baro", "gps", "odom", "flow"}


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


def true_gyro_bias_from_sensor_noise(sensor_noise: EKFSensorNoise) -> np.ndarray:
    turn_on_bias = np.asarray(
        getattr(sensor_noise, "gyro_turn_on_bias", np.zeros(3, dtype=float)),
        dtype=float,
    ).reshape(3)
    colored_bias = np.asarray(
        getattr(sensor_noise, "gyro_bias", np.zeros(3, dtype=float)),
        dtype=float,
    ).reshape(3)
    return turn_on_bias + colored_bias


def format_dropout_schedule() -> str:
    return ", ".join(
        f"{phase_name}[{occurrence}] @ +{start:.1f}s to +{start + duration:.1f}s"
        for phase_name, occurrence, start, duration in AUTO_DROPOUT_PHASE_WINDOWS
    )


def find_phase_by_name(
    mission: FlightMission,
    phase_name: str,
    occurrence: int,
):
    matches = [phase for phase in mission.phases if str(phase.name) == str(phase_name)]
    if 0 <= int(occurrence) < len(matches):
        return matches[int(occurrence)]
    return None


def absolute_dropout_windows(mission: FlightMission) -> list[tuple[float, float]]:
    windows = []
    for phase_name, occurrence, start_s, duration_s in AUTO_DROPOUT_PHASE_WINDOWS:
        phase = find_phase_by_name(mission, phase_name, occurrence)
        if phase is not None:
            windows.append((float(phase.t_start + start_s), float(duration_s)))
    return windows


def get_active_dropout_phase_windows(disable_dropout: bool) -> tuple[tuple[str, int, float, float], ...]:
    if disable_dropout:
        return ()
    return AUTO_DROPOUT_PHASE_WINDOWS


def mission_plan_lines() -> list[str]:
    return [
        "Mission plan:",
        "  1. Takeoff: 0.0s to 2.0s -> climb to 1.0 m",
        "  2. Hover: 2.0s to 3.0s -> settle before trajectory tracking",
        "  3. Square: 3.0s to 15.0s -> main trajectory with split dropout windows",
        "     square dropout 1: +1.0s to +3.5s",
        "     square dropout 2: +8.5s to +11.0s",
        "  4. Hover: 15.0s to 16.0s -> settle after trajectory",
        "  5. Landing: 16.0s to 20.0s -> descend to ground",
    ]


def update_auto_dropout(
    mission: FlightMission,
    phase_windows: tuple[tuple[str, int, float, float], ...],
    t: float,
    next_window_idx: int,
    auto_active: bool,
) -> tuple[int, bool]:
    """Run the configured scheduled GPS-only dropout windows."""
    if next_window_idx >= len(phase_windows):
        return next_window_idx, auto_active

    current_phase = mission.phase_at(t)
    target_phase_name, target_occurrence, start_s, duration_s = phase_windows[next_window_idx]
    target_phase = find_phase_by_name(mission, target_phase_name, target_occurrence)
    end_s = start_s + duration_s
    if (
        not auto_active
        and current_phase is not None
        and target_phase is not None
        and current_phase is target_phase
        and start_s <= (t - current_phase.t_start) < end_s
    ):
        return next_window_idx, True

    if auto_active and (
        current_phase is None
        or target_phase is None
        or current_phase is not target_phase
        or (t - current_phase.t_start) >= end_s
    ):
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


def scale_measurement_component(
    component: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, H, R = component
    return (
        np.asarray(z, dtype=float),
        np.asarray(H, dtype=float),
        np.asarray(R, dtype=float) * float(scale),
    )


def build_mission():
    mission = FlightMission(default_z=1.0, ground_z=0.0)
    mission.add_takeoff(duration=2.0, target_z=1.0)
    mission.add_hover(duration=1.0, z=1.0)
    mission.add_square(
        duration=12.0,
        side=2.0,
        period=12.0,
        z=1.0,
        offset_xy=(0.0, 0.0),
    )
    mission.add_hover(duration=1.0, z=1.0)
    mission.add_landing(duration=4.0, ground_z=0.055)
    return mission


def hover_omega_from_ekf_params(ekf: PhoenixEKFAdapter) -> np.ndarray:
    params = ekf.ekf.params
    return np.full(4, float(np.sqrt(params.m * params.g / (4.0 * params.b))), dtype=float)


def get_optional_dropout_measurements(env) -> dict:
    drone = getattr(env, "drone", None)
    odom_position = getattr(env, "odometry_position", None)
    odom_velocity = getattr(env, "odometry_velocity", None)
    flow_velocity_xy = getattr(env, "optical_flow_velocity_xy", None)

    if drone is not None:
        if odom_position is None:
            odom_position = getattr(drone, "odometry_position", None)
        if odom_velocity is None:
            odom_velocity = getattr(drone, "odometry_velocity", None)
        if flow_velocity_xy is None:
            flow_velocity_xy = getattr(drone, "optical_flow_velocity_xy", None)

    return {
        "odom_position": None if odom_position is None else np.asarray(odom_position, dtype=float).reshape(3),
        "odom_velocity": None if odom_velocity is None else np.asarray(odom_velocity, dtype=float).reshape(3),
        "optical_flow_velocity_xy": (
            None
            if flow_velocity_xy is None
            else np.asarray(flow_velocity_xy, dtype=float).reshape(2)
        ),
    }


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
    parser.add_argument(
        "--no-dropout",
        action="store_true",
        help="Disable the scheduled dropout windows for this run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mission = build_mission()
    active_dropout_phase_windows = get_active_dropout_phase_windows(args.no_dropout)
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

    ekf_model = QuadcopterEKF(
        dt=env.TIME_STEP,
        sigma_gyro_bias=EKF_SIGMA_GYRO_BIAS,
        q_bias_floor=EKF_Q_BIAS_FLOOR,
        initial_cov_diag=np.array(
            [
                0.05, 0.05, 0.05,
                0.10, 0.10, 0.10,
                0.05, 0.05, 0.05,
                0.10, 0.10, 0.10,
                EKF_INITIAL_GYRO_BIAS_STD,
                EKF_INITIAL_GYRO_BIAS_STD,
                EKF_INITIAL_GYRO_BIAS_STD,
            ],
            dtype=float,
        ),
    )
    ekf = PhoenixEKFAdapter(
        dt=env.TIME_STEP,
        ekf=ekf_model,
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
    state_dim = int(ekf.ekf.x.size)
    omega_hover = hover_omega_from_ekf_params(ekf)

    print("Mission started with EKF state feedback.")
    if args.render:
        print("Keys: H=hover, R=return-home, L=land, C=clear dropout")
        if active_dropout_phase_windows:
            print(f"Automatic hover dropout windows: {format_dropout_schedule()}")
        else:
            print("Automatic hover dropout windows: disabled")
    else:
        print("Running headless. Use --render to enable the PyBullet GUI and keyboard dropout controls.")
    for line in mission_plan_lines():
        print(line)
    dropout_windows_abs = absolute_dropout_windows(mission) if active_dropout_phase_windows else []

    last_ctrl_pos = env.drone.xyz.copy()
    noise_gen     = EKFSensorNoise(
        sample_turn_on_bias_once=True,
        gyro_turn_on_bias_sigma=0.0,
    )

    steps = int(mission.total_time / env.TIME_STEP)
    log_t = []
    log_ref = []
    log_ref_vel = []
    log_estimate = []
    log_estimate_vel = []
    log_measurement = []
    log_measurement_vel = []
    log_true = []
    log_true_vel = []
    log_nees = []
    log_nees_trans = []
    log_nees_att_rate = []
    log_nis_real_norm = []
    log_nis_nav_norm = []
    log_nis_inertial_norm = []
    log_nis_pseudo_norm = []
    log_dropout_active = []
    log_dropout_duration = []
    log_measurement_dim_real = []
    log_measurement_dim_pseudo = []
    log_sensor_nis = {name: [] for name in REAL_SENSOR_NAMES}
    prev_gps_dropout_active = False
    prev_mission_override_active = False
    gps_recovery_time_left = 0.0
    hold_ctrl_pos = np.asarray(env.drone.xyz, dtype=float).copy()
    trail_prev_pos = np.asarray(env.drone.xyz, dtype=float).copy()
    prev_ref_pos = np.asarray(env.get_mission_reference(), dtype=float).copy()
    last_valid_ref_pos = prev_ref_pos.copy()
    last_valid_ref_vel = np.zeros(3, dtype=float)
    dropout_hold_pos = prev_ref_pos.copy()
    recovery_anchor_pos = prev_ref_pos.copy()
    recovery_time_left = 0.0
    auto_dropout_next_idx = 0
    auto_dropout_active = False
    for k in range(steps):
        if args.render:
            time.sleep(env.TIME_STEP)
        wall_time = float(k * env.TIME_STEP)

        auto_dropout_next_idx, auto_dropout_active = update_auto_dropout(
            mission,
            active_dropout_phase_windows,
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

        mission_override_active = bool(env.dropout_mgr.active)
        dropout_mode = env.dropout_mgr.mode if mission_override_active else "NONE"
        gps_dropout_active = bool(auto_dropout_active or mission_override_active)
        if gps_dropout_active and not prev_gps_dropout_active:
            dropout_hold_pos = np.asarray(estimate["x"], dtype=float).copy()
            dropout_hold_pos[2] = float(last_valid_ref_pos[2])
            ekf.ekf.P[0:2, 0:2] *= 160.0
            ekf.ekf.P[2, 2] *= 16.0
            ekf.ekf.P[3:5, 3:5] *= 96.0
            ekf.ekf.P[5, 5] *= 12.0
            reset_pid_integrators(quad)
        if mission_override_active and not prev_mission_override_active:
            hold_ctrl_pos = np.asarray(env.get_mission_reference(), dtype=float).copy()
            prev_ref_pos = hold_ctrl_pos.copy()
            last_ctrl_pos = hold_ctrl_pos.copy()
            reset_pid_integrators(quad)
        elif not mission_override_active and prev_mission_override_active:
            reset_pid_integrators(quad)
        if not gps_dropout_active and prev_gps_dropout_active:
            gps_recovery_time_left = GPS_RECOVERY_GATE_DURATION_S
            recovery_anchor_pos = np.asarray(estimate["x"], dtype=float).copy()
            recovery_time_left = RECOVERY_BLEND_DURATION_S

        if mission_override_active:
            pos_ref = np.asarray(env.get_mission_reference(), dtype=float)
            vel_ref = np.zeros(3, dtype=float)
        else:
            pos_ref, vel_ref = env.current_reference()
            pos_ref = np.asarray(pos_ref, dtype=float)
            vel_ref = np.asarray(vel_ref, dtype=float)

        if not gps_dropout_active and not mission_override_active:
            last_valid_ref_pos = pos_ref.copy()
            last_valid_ref_vel = vel_ref.copy()

        if gps_dropout_active and not mission_override_active:
            pos_ref[:2] = dropout_hold_pos[:2]
            vel_ref[:2] = 0.0
            vel_ref[2] = 0.0
        elif (
            not gps_dropout_active
            and not mission_override_active
            and recovery_time_left > 0.0
        ):
            alpha = 1.0 - recovery_time_left / RECOVERY_BLEND_DURATION_S
            alpha = float(np.clip(alpha, 0.0, 1.0))
            pos_ref[:2] = (1.0 - alpha) * recovery_anchor_pos[:2] + alpha * pos_ref[:2]
            vel_ref[:2] = alpha * vel_ref[:2]
            recovery_time_left = max(0.0, recovery_time_left - env.TIME_STEP)

        ekf_dropout_active = gps_dropout_active
        if mission_override_active and dropout_mode == "HOV":
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
        elif mission_override_active and dropout_mode in {"RTH", "LAND"}:
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
        ctrl = quad.step(
            pos_ref,
            vel_ref,
            z_ref=z_ref,
            freeze_z_integrator=True,
            control_profile="dropout" if gps_dropout_active else "nominal",
        )

        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)
        if args.render:
            add_trail_segment(
                env,
                trail_prev_pos,
                np.asarray(env.drone.xyz, dtype=float),
                dropout_active=gps_dropout_active,
            )
        trail_prev_pos = np.asarray(env.drone.xyz, dtype=float).copy()
        omega_applied = motor_omega_from_applied_forces(env, ekf)
        if ekf_dropout_active:
            ekf.dropout_time += env.TIME_STEP
            omega_predict = omega_applied
            if mission_override_active and dropout_mode == "HOV":
                omega_predict = omega_hover
            ekf.ekf.predict_dropout(
                omega=omega_predict,
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
            ekf.ekf.predict(omega=omega_applied, dt=env.TIME_STEP)

        noisy_pos, noisy_vel, noisy_att, noisy_rates, _ = ekf.sensor_noise.add_noise(
            np.asarray(env.drone.xyz, dtype=float),
            np.asarray(env.drone.xyz_dot, dtype=float),
            np.asarray(env.drone.rpy, dtype=float),
            np.asarray(env.drone.rpy_dot, dtype=float),
            np.zeros(3, dtype=float),
            env.TIME_STEP,
        )
        baro_z = ekf.sensor_noise.add_noise_to_baro(float(env.drone.xyz[2]))
        optional_meas = get_optional_dropout_measurements(env)
        noisy_state = {
            "position": noisy_pos,
            "velocity": noisy_vel,
            "attitude": noisy_att,
            "rates": noisy_rates,
            "baro_z": float(baro_z),
            "odom_position": optional_meas["odom_position"],
            "odom_velocity": optional_meas["odom_velocity"],
            "optical_flow_velocity_xy": optional_meas["optical_flow_velocity_xy"],
            "dropout_duration_s": float(ekf.dropout_time),
            "dropout_active": bool(ekf_dropout_active),
        }

        nis_real_terms = []
        nis_nav_terms = []
        nis_inertial_terms = []
        nis_pseudo_terms = []
        measurement_dim_real = 0
        measurement_dim_nav = 0
        measurement_dim_inertial = 0
        measurement_dim_pseudo = 0
        step_sensor_nis = {name: np.nan for name in REAL_SENSOR_NAMES}

        def _apply_update(
            component: tuple[np.ndarray, np.ndarray, np.ndarray],
            *,
            gate_threshold: float | None = None,
            is_pseudo: bool = False,
            sensor_name: str | None = None,
        ) -> bool:
            nonlocal measurement_dim_real, measurement_dim_nav, measurement_dim_inertial, measurement_dim_pseudo
            z_u, H_u, R_u = component
            innovation_u = ekf.ekf.innovation(z_u, H_u)
            S_u = H_u @ ekf.ekf.P @ H_u.T + R_u
            nis_u = compute_nis(innovation_u, S_u)
            accepted = ekf.ekf.update(
                measurement=z_u,
                H=H_u,
                measurement_noise=R_u,
                gate_threshold=gate_threshold,
            )
            if accepted:
                meas_dim = int(np.asarray(z_u, dtype=float).size)
                if is_pseudo:
                    nis_pseudo_terms.append(nis_u)
                    measurement_dim_pseudo += meas_dim
                else:
                    nis_real_terms.append(nis_u)
                    measurement_dim_real += meas_dim
                    if sensor_name is not None:
                        step_sensor_nis[str(sensor_name)] = nis_u / float(max(1, meas_dim))
                        if str(sensor_name) in NAV_SENSOR_NAMES:
                            nis_nav_terms.append(nis_u)
                            measurement_dim_nav += meas_dim
                        else:
                            nis_inertial_terms.append(nis_u)
                            measurement_dim_inertial += meas_dim
            return bool(accepted)

        if ekf_dropout_active:
            dropout_components = [
                (
                    scale_measurement_component(
                        ekf.ekf.build_attitude_measurement(noisy_att),
                        ATTITUDE_R_SCALE,
                    ),
                    False,
                    "attitude",
                ),
                (
                    scale_measurement_component(
                        ekf.ekf.build_gyro_measurement(noisy_rates),
                        GYRO_R_SCALE,
                    ),
                    False,
                    "gyro",
                ),
                (
                    ekf.ekf.build_velocity_pseudo_measurement(
                        vel_ref,
                        std_xy=PSEUDO_VEL_STD_XY,
                        std_z=PSEUDO_VEL_STD_Z,
                    ),
                    True,
                    None,
                ),
                (
                    scale_measurement_component(
                        ekf.ekf.build_baro_measurement(
                            baro_z,
                            std=ekf.sensor_noise.baro_noise_std,
                        ),
                        BARO_R_SCALE,
                    ),
                    False,
                    "baro",
                ),
            ]
            if (
                optional_meas["odom_position"] is not None
                or optional_meas["odom_velocity"] is not None
            ):
                odom_position = noisy_pos if optional_meas["odom_position"] is None else optional_meas["odom_position"]
                odom_velocity = noisy_vel if optional_meas["odom_velocity"] is None else optional_meas["odom_velocity"]
                dropout_components.append(
                    (
                        scale_measurement_component(
                            ekf.ekf.build_odom_measurement(
                                position=odom_position,
                                velocity=odom_velocity,
                            ),
                            ODOM_R_SCALE,
                        ),
                        False,
                        "odom",
                    )
                )
            elif optional_meas["optical_flow_velocity_xy"] is not None:
                dropout_components.append(
                    (
                        scale_measurement_component(
                            ekf.ekf.build_optical_flow_measurement(
                                optional_meas["optical_flow_velocity_xy"]
                            ),
                            FLOW_R_SCALE,
                        ),
                        False,
                        "flow",
                    )
                )
            for component, is_pseudo, sensor_name in dropout_components:
                _apply_update(component, is_pseudo=is_pseudo, sensor_name=sensor_name)
        else:
            nominal_components = [
                (
                    scale_measurement_component(
                        ekf.ekf.build_attitude_measurement(noisy_att),
                        ATTITUDE_R_SCALE,
                    ),
                    False,
                    "attitude",
                ),
                (
                    scale_measurement_component(
                        ekf.ekf.build_gyro_measurement(noisy_rates),
                        GYRO_R_SCALE,
                    ),
                    False,
                    "gyro",
                ),
                (
                    scale_measurement_component(
                        ekf.ekf.build_velocity_measurement(noisy_vel),
                        VELOCITY_R_SCALE,
                    ),
                    False,
                    "velocity",
                ),
                (
                    scale_measurement_component(
                        ekf.ekf.build_baro_measurement(
                            baro_z,
                            std=ekf.sensor_noise.baro_noise_std,
                        ),
                        BARO_R_SCALE,
                    ),
                    False,
                    "baro",
                ),
            ]
            for component, is_pseudo, sensor_name in nominal_components:
                _apply_update(component, is_pseudo=is_pseudo, sensor_name=sensor_name)

            gps_gate_threshold = (
                GPS_RECOVERY_GATE_THRESHOLD if gps_recovery_time_left > 0.0 else None
            )
            _apply_update(
                scale_measurement_component(
                    ekf.ekf.build_gps_measurement(noisy_pos),
                    GPS_R_SCALE,
                ),
                gate_threshold=gps_gate_threshold,
                is_pseudo=False,
                sensor_name="gps",
            )
            gps_recovery_time_left = max(0.0, gps_recovery_time_left - env.TIME_STEP)

        nis_real_norm = sum(nis_real_terms) / float(max(1, measurement_dim_real))
        nis_nav_norm = sum(nis_nav_terms) / float(max(1, measurement_dim_nav))
        nis_inertial_norm = sum(nis_inertial_terms) / float(max(1, measurement_dim_inertial))
        nis_pseudo_norm = sum(nis_pseudo_terms) / float(max(1, measurement_dim_pseudo))

        estimate = ekf.ekf.as_dict()
        estimate["measurement"] = noisy_state
        x_true = np.concatenate(
            [
                np.asarray(env.drone.xyz, dtype=float),
                np.asarray(env.drone.xyz_dot, dtype=float),
                np.asarray(env.drone.rpy, dtype=float),
                np.asarray(env.drone.rpy_dot, dtype=float),
                true_gyro_bias_from_sensor_noise(ekf.sensor_noise),
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
        prev_gps_dropout_active = gps_dropout_active
        prev_mission_override_active = mission_override_active
        estimated_pos = np.asarray(estimate["x"], dtype=float)
        measured_pos = np.asarray(estimate["measurement"]["position"], dtype=float)
        true_pos = np.asarray(env.drone.xyz, dtype=float)
        sim_time = float((k + 1) * env.TIME_STEP)
        log_t.append(sim_time)
        log_ref.append(pos_ref.copy())
        log_ref_vel.append(np.asarray(vel_ref, dtype=float).copy())
        log_estimate.append(estimated_pos.copy())
        log_estimate_vel.append(np.asarray(estimate["v"], dtype=float).copy())
        log_measurement.append(measured_pos.copy())
        log_measurement_vel.append(np.asarray(estimate["measurement"]["velocity"], dtype=float).copy())
        log_true.append(true_pos.copy())
        log_true_vel.append(np.asarray(env.drone.xyz_dot, dtype=float).copy())
        log_nees.append(nees)
        log_nees_trans.append(nees_trans)
        log_nees_att_rate.append(nees_att_rate)
        log_nis_real_norm.append(nis_real_norm)
        log_nis_nav_norm.append(nis_nav_norm)
        log_nis_inertial_norm.append(nis_inertial_norm)
        log_nis_pseudo_norm.append(nis_pseudo_norm)
        log_dropout_active.append(gps_dropout_active)
        log_dropout_duration.append(float(ekf.dropout_time))
        log_measurement_dim_real.append(int(measurement_dim_real))
        log_measurement_dim_pseudo.append(int(measurement_dim_pseudo))
        for sensor_name in REAL_SENSOR_NAMES:
            log_sensor_nis[sensor_name].append(float(step_sensor_nis[sensor_name]))
        prev_ref_pos = pos_ref.copy()

        if k % 20 == 0:
            pos = estimate["x"]
            print(
                f"t={sim_time:6.2f}s  ref=({pos_ref[0]: .2f},{pos_ref[1]: .2f},{pos_ref[2]: .2f})  "
                f"est=({pos[0]: .2f},{pos[1]: .2f},{pos[2]: .2f})  "
                f"dropout={'ON' if gps_dropout_active else 'OFF'}  "
                f"drop_t={ekf.dropout_time:4.2f}s"
            )

        if done or truncated:
            print("Mission terminated.")
            break

    env.close()

    if log_t:
        t_arr = np.asarray(log_t, dtype=float)
        ref_arr = np.asarray(log_ref, dtype=float)
        ref_vel_arr = np.asarray(log_ref_vel, dtype=float)
        estimate_arr = np.asarray(log_estimate, dtype=float)
        estimate_vel_arr = np.asarray(log_estimate_vel, dtype=float)
        measurement_arr = np.asarray(log_measurement, dtype=float)
        measurement_vel_arr = np.asarray(log_measurement_vel, dtype=float)
        true_arr = np.asarray(log_true, dtype=float)
        true_vel_arr = np.asarray(log_true_vel, dtype=float)
        nees_arr = np.asarray(log_nees, dtype=float)
        nees_trans_arr = np.asarray(log_nees_trans, dtype=float)
        nees_att_rate_arr = np.asarray(log_nees_att_rate, dtype=float)
        nis_real_arr = np.asarray(log_nis_real_norm, dtype=float)
        nis_nav_arr = np.asarray(log_nis_nav_norm, dtype=float)
        nis_inertial_arr = np.asarray(log_nis_inertial_norm, dtype=float)
        nis_pseudo_arr = np.asarray(log_nis_pseudo_norm, dtype=float)
        dropout_mask = np.asarray(log_dropout_active, dtype=bool)
        nominal_mask = ~dropout_mask
        measurement_dim_real_arr = np.asarray(log_measurement_dim_real, dtype=int)
        measurement_dim_pseudo_arr = np.asarray(log_measurement_dim_pseudo, dtype=int)
        sensor_nis_arr = {
            sensor_name: np.asarray(values, dtype=float)
            for sensor_name, values in log_sensor_nis.items()
        }
        pos_err_vec = estimate_arr - true_arr
        pos_err_norm = np.linalg.norm(pos_err_vec, axis=1)
        lateral_err_norm = np.linalg.norm(pos_err_vec[:, :2], axis=1)
        altitude_err_abs = np.abs(pos_err_vec[:, 2])
        pos_err_ref_norm = np.linalg.norm(estimate_arr - ref_arr, axis=1)

        if not args.no_plot:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axis_labels = ["X [m]", "Y [m]", "Z [m]"]
            for i in range(3):
                for start_s, duration_s in dropout_windows_abs:
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

            fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            vel_axis_labels = ["Vx [m/s]", "Vy [m/s]", "Vz [m/s]"]
            for i in range(3):
                for start_s, duration_s in dropout_windows_abs:
                    axs_vel[i].axvspan(
                        start_s,
                        start_s + duration_s,
                        color="0.9",
                        alpha=0.5,
                        zorder=0,
                    )
                nominal_series = np.where(nominal_mask, estimate_vel_arr[:, i], np.nan)
                dropout_series = np.where(dropout_mask, estimate_vel_arr[:, i], np.nan)
                axs_vel[i].plot(t_arr, ref_vel_arr[:, i], "--", color="black", linewidth=1.8, label="reference")
                axs_vel[i].plot(
                    t_arr,
                    true_vel_arr[:, i],
                    color="0.4",
                    linewidth=1.1,
                    label="true velocity",
                )
                axs_vel[i].plot(
                    t_arr,
                    measurement_vel_arr[:, i],
                    color="0.7",
                    alpha=0.9,
                    linewidth=1.0,
                    label="noisy measurement",
                )
                axs_vel[i].plot(t_arr, nominal_series, color=TRAIL_BLUE, linewidth=1.8, label="EKF estimate (nominal)")
                axs_vel[i].plot(t_arr, dropout_series, color=TRAIL_RED, linewidth=1.8, label="EKF estimate (dropout)")
                axs_vel[i].set_ylabel(vel_axis_labels[i])
                axs_vel[i].grid(True, alpha=0.3)
                axs_vel[i].legend(loc="best")

            axs_vel[-1].set_xlabel("Time [s]")
            fig_vel.suptitle("Velocity Reference, Truth, EKF Estimate, and Noisy Measurement")
            fig_vel.tight_layout()

            fig_consistency, axs_consistency = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for ax in axs_consistency:
                for start_s, duration_s in dropout_windows_abs:
                    ax.axvspan(
                        start_s,
                        start_s + duration_s,
                        color="0.9",
                        alpha=0.5,
                        zorder=0,
                    )
            axs_consistency[0].plot(t_arr, nees_arr, linewidth=1.5, label="NEES")
            axs_consistency[0].axhline(
                state_dim,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"expected = {state_dim}",
            )
            axs_consistency[0].set_ylabel("NEES")
            axs_consistency[0].grid(True, alpha=0.3)
            axs_consistency[0].legend(loc="best")

            axs_consistency[1].plot(t_arr, nis_nav_arr, linewidth=1.5, label="NIS_nav / DOF")
            axs_consistency[1].plot(
                t_arr,
                nis_inertial_arr,
                linewidth=1.2,
                linestyle="--",
                color="tab:green",
                label="NIS_inertial / DOF",
            )
            axs_consistency[1].plot(
                t_arr,
                nis_pseudo_arr,
                linewidth=1.2,
                linestyle=":",
                color="tab:orange",
                label="NIS_pseudo / DOF",
            )
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
            mean_nis_real = float(np.mean(nis_real_arr[mask]))
            mean_nis_nav = float(np.mean(nis_nav_arr[mask]))
            mean_nis_inertial = float(np.mean(nis_inertial_arr[mask]))
            mean_nis_pseudo = float(np.mean(nis_pseudo_arr[mask]))
            mean_meas_dim_real = float(np.mean(measurement_dim_real_arr[mask]))
            mean_meas_dim_pseudo = float(np.mean(measurement_dim_pseudo_arr[mask]))
            print(
                f"{label}: duration={duration_s:.2f}s  samples={sample_count}  "
                f"mean NEES={mean_nees:.2f}  mean NIS_real/DOF={mean_nis_real:.3f}  "
                f"mean NIS_nav/DOF={mean_nis_nav:.3f}  "
                f"mean NIS_inertial/DOF={mean_nis_inertial:.3f}  "
                f"mean NIS_pseudo/DOF={mean_nis_pseudo:.3f}  "
                f"mean real measurement dim={mean_meas_dim_real:.1f}  "
                f"mean pseudo measurement dim={mean_meas_dim_pseudo:.1f}"
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

        def print_lateral_altitude_block(label: str, mask: np.ndarray) -> None:
            sample_count = int(np.count_nonzero(mask))
            if sample_count == 0:
                print(f"{label}: no samples")
                return
            lateral_rmse = float(np.sqrt(np.mean(lateral_err_norm[mask] ** 2)))
            lateral_max = float(np.max(lateral_err_norm[mask]))
            altitude_rmse = float(np.sqrt(np.mean(altitude_err_abs[mask] ** 2)))
            altitude_max = float(np.max(altitude_err_abs[mask]))
            print(
                f"{label}: lateral RMSE={lateral_rmse:.3f} m  "
                f"lateral max={lateral_max:.3f} m  "
                f"altitude RMSE={altitude_rmse:.3f} m  "
                f"altitude max={altitude_max:.3f} m"
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
            if not dropout_windows_abs:
                return
            for idx, (start_s, duration_s) in enumerate(dropout_windows_abs, start=1):
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

        def print_sensor_nis_block(label: str, mask: np.ndarray) -> None:
            print(label + ":")
            for sensor_name in REAL_SENSOR_NAMES:
                arr = sensor_nis_arr[sensor_name]
                valid = mask & np.isfinite(arr)
                count = int(np.count_nonzero(valid))
                if count == 0:
                    print(f"  {sensor_name}: no accepted samples")
                    continue
                mean_nis = float(np.mean(arr[valid]))
                print(f"  {sensor_name}: mean NIS/DOF={mean_nis:.3f}  samples={count}")

        print(
            f"Mean consistency metrics: NEES={nees_arr.mean():.2f} "
            f"(expected {state_dim}), "
            f"NIS_real/DOF={nis_real_arr.mean():.3f}, "
            f"NIS_nav/DOF={nis_nav_arr.mean():.3f}, "
            f"NIS_inertial/DOF={nis_inertial_arr.mean():.3f}, "
            f"NIS_pseudo/DOF={nis_pseudo_arr.mean():.3f}"
        )
        print_consistency_block("Nominal only", nominal_mask)
        print_consistency_block("Dropout only", dropout_mask)
        print_group_nees_block("Nominal grouped NEES", nominal_mask)
        print_group_nees_block("Dropout grouped NEES", dropout_mask)
        print_position_block("Nominal position errors", nominal_mask)
        print_position_block("Dropout position errors", dropout_mask)
        print_lateral_altitude_block("Nominal lateral/altitude errors", nominal_mask)
        print_lateral_altitude_block("Dropout lateral/altitude errors", dropout_mask)
        print_sensor_nis_block("Nominal per-sensor NIS", nominal_mask)
        print_sensor_nis_block("Dropout per-sensor NIS", dropout_mask)
        print_recovery_block()
        if not args.no_plot:
            plt.show()


if __name__ == "__main__":
    main()
