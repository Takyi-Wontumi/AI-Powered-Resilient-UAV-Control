#!/usr/bin/env python3
"""
Adaptive Search Mission - Pre-Compensation Hybrid Model
========================================================
Uses LSTM multi-horizon drift prediction to pre-compensate waypoints
BEFORE the drone reaches them, rather than correcting EKF position.

Each execution generates a DIFFERENT randomized search pattern.
GPS dropout windows demonstrate pre-compensation mode.
"""

import sys, os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for path in [
    SCRIPT_DIR,
    ROOT_DIR,
    os.path.join(ROOT_DIR, 'AI_UAV_Tests'),
    os.path.join(ROOT_DIR, 'GPS_Dropout_Recovery'),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

print("\n" + "="*80)
print("ADAPTIVE SEARCH MISSION - PRE-COMPENSATION HYBRID MODEL")
print("="*80)
print("\nUsing KALMAN FILTER + LSTM WAYPOINT PRE-COMPENSATION")
print("Real-time PyBullet visualization with fixed square mission\n")

# =========================================================
# Pre-compensation constants
# =========================================================
SHADOW_RESET_INTERVAL_PRECOMP = 2500   # must match data collector
HORIZON_STRIDE_PRECOMP        = 100    # 100 steps x 0.005s = 0.5s per horizon
NUM_HORIZONS_PRECOMP           = 7     # +0.5s .. +3.5s
MAX_PRECOMP_HORIZON_S          = 3.5   # max LSTM coverage in seconds
PSEUDO_VEL_STD_XY              = 2.10
PSEUDO_VEL_STD_Z               = 4.00
EKF_SIGMA_GYRO_BIAS            = 0.001
EKF_Q_BIAS_FLOOR               = 1.0e-8
EKF_INITIAL_GYRO_BIAS_STD      = 0.001
PRECOMP_DELAY_S                = 0.5
MAX_PRECOMP_SHIFT              = 1.50
PRECOMP_INITIAL_SHIFT_CAP      = 0.03
PRECOMP_RAMP_DURATION_S        = 1.0
MAX_SHIFT_DELTA_PER_UPDATE     = 0.060
PRECOMP_MIN_ANCHOR_DRIFT_M     = 0.10
PRECOMP_FULL_ANCHOR_DRIFT_M    = 0.30
PRECOMP_WAYPOINT_FRACTION      = 0.60
SQUARE_SIDE_X                  = 1.5
SQUARE_SIDE_Y                  = 1.5
SQUARE_POINTS_PER_EDGE         = 6
DEMO_TARGET_SPEED              = 0.8
DEMO_DROPOUT_DURATION_S        = 2.0
DEMO_NUM_DROPOUTS              = 1
EKF_INITIAL_COV_DIAG = np.array(
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
)

# =========================================================
# Load Pre-Compensation LSTM Model
# =========================================================
print("[1] Loading pre-compensation drift model...")

class PositionDriftLSTMPrecomp(nn.Module):
    """7-input (vel+gyro+t_norm), 7-horizon LSTM for waypoint pre-compensation."""
    def __init__(self, horizon_steps=7, hidden_size=256):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.proj = nn.Linear(7, hidden_size)  # vel(3)+gyro(3)+t_norm(1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1,
                            batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3)
            )
            for _ in range(horizon_steps)
        ])

    def forward(self, x):
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        outputs = [head(final) for head in self.heads]
        return torch.stack(outputs, dim=1)  # (B, 7, 3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PositionDriftLSTMPrecomp(7, 256).to(device)

_precomp_model = Path('experiments/lstm_position_drift_precomp.pt')

if _precomp_model.exists():
    model.load_state_dict(torch.load(_precomp_model, map_location=device))
    model.eval()
    print("    [OK] Pre-compensation LSTM loaded (lstm_position_drift_precomp.pt)")
    print("    Input: vel+gyro+t_norm | Target: position drift at +0.5s..+3.5s\n")
else:
    print("[ERROR] No pre-compensation model found.")
    print("  Run: python collect_pybullet_position_drift_data_precomp.py")
    print("  Then: python train_lstm_position_drift_precomp.py")
    sys.exit(1)
# (model load is now handled above; sys.exit on missing model)

# =========================================================
# Setup Environment
# =========================================================
print("[2] Creating PyBullet environment with GUI...")

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import QuadcopterEKF
from AI_UAV_Tests.sensors_ekf import EKFSensorNoise

def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


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


def motor_omega_from_applied_forces(env, ekf_model: QuadcopterEKF) -> np.ndarray:
    motor_forces = np.asarray(env.drone.y, dtype=float).reshape(4)
    thrust_coeff = float(ekf_model.params.b)
    return np.sqrt(np.clip(motor_forces, 0.0, np.inf) / thrust_coeff)

env = DroneMissionEnv(
    physics="PyBulletPhysics",
    control_mode="AttitudeRate",
    drone_model="cf21x_bullet",
    dropout_mode="NONE",
    render_mode="human"
)

env.drone.control = AttitudeRate(
    bc=env.bc,
    drone=env.drone,
    time_step=env.TIME_STEP
)

obs, info = env.reset()
print("    [OK] Environment ready with PyBullet GUI\n")

# =========================================================
# Tracer Drawing Functions
# =========================================================
def draw_line(bc, start, end, color, width=2):
    """Draw a line in PyBullet"""
    bc.addUserDebugLine(
        lineFromXYZ=start,
        lineToXYZ=end,
        lineColorRGB=color,
        lineWidth=width
    )

def create_wp_sphere(bc, pos, radius=0.06, rgba=(0.5, 0.5, 0.5, 0.7)):
    """Create a visual-only (no collision) sphere. Returns body_id."""
    vis = bc.createVisualShape(bc.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba))
    return bc.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                              basePosition=list(pos))

def move_sphere(bc, body_id, pos):
    """Teleport an existing sphere body to a new XYZ position."""
    bc.resetBasePositionAndOrientation(body_id, list(pos), [0, 0, 0, 1])

def recolor_sphere(bc, body_id, rgba):
    """Change the RGBA colour of an existing sphere."""
    bc.changeVisualShape(body_id, -1, rgbaColor=list(rgba))

bc_client = env.bc

# =========================================================
# Waypoint/Search Pattern Generation
# =========================================================
print("[3] Generating fixed square mission...\n")

class AdaptiveSearchPatternGenerator:
    """Generate waypoint patterns for the demo mission."""
    
    def __init__(self, area_size=(10.0, 5.0), altitude=1.0):
        self.x_min, self.x_max = -area_size[0]/2, area_size[0]/2
        self.y_min, self.y_max = -area_size[1]/2, area_size[1]/2
        self.z = altitude
        self.area_size = area_size
        
    def square_search(self, side_x=2.0, side_y=2.0, points_per_edge=12):
        waypoints = []
        cx, cy = 0.0, 0.0
        corners = np.array([
            [cx - side_x / 2.0, cy - side_y / 2.0, self.z],
            [cx + side_x / 2.0, cy - side_y / 2.0, self.z],
            [cx + side_x / 2.0, cy + side_y / 2.0, self.z],
            [cx - side_x / 2.0, cy + side_y / 2.0, self.z],
        ], dtype=float)

        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            for j in range(points_per_edge):
                alpha = j / points_per_edge
                pt = (1.0 - alpha) * start + alpha * end
                pt[0] = np.clip(pt[0], self.x_min, self.x_max)
                pt[1] = np.clip(pt[1], self.y_min, self.y_max)
                waypoints.append(pt.copy())

        return np.array(waypoints)

generator = AdaptiveSearchPatternGenerator(area_size=(3.5, 2.5), altitude=1.0)
pattern_type = 'square'
print(f"Generated pattern: {pattern_type.upper()}")
waypoints = generator.square_search(
    side_x=SQUARE_SIDE_X,
    side_y=SQUARE_SIDE_Y,
    points_per_edge=SQUARE_POINTS_PER_EDGE,
)

print(f"Generated {len(waypoints)} waypoints\n")

# =========================================================
# Mission Configuration
# =========================================================
quad = QuadcopterPID(dt=env.TIME_STEP)
quad.reset()

# ── Waypoint spheres ─────────────────────────────────────────────────────────
# Grey   (small)  : original planned positions — fixed throughout mission
# Green  (medium) : LSTM-adjusted targets, only next N_SHOW ahead visible
# Yellow (large)  : current active waypoint being tracked
N_SHOW = 4   # how many upcoming green spheres to show at once
_BURIED = [0.0, 0.0, -99.0]   # park hidden spheres underground
print("[3b] Creating waypoint visualisation spheres...")
orig_spheres = [create_wp_sphere(bc_client, wp, radius=0.05, rgba=(0.7, 0.7, 0.7, 0.6))
                for wp in waypoints]
# Green spheres all start buried; refresh_green_window() surfaces the right ones
adj_spheres  = [create_wp_sphere(bc_client, _BURIED, radius=0.07, rgba=(0.1, 0.9, 0.1, 0.9))
                for _ in waypoints]
curr_sphere  = create_wp_sphere(bc_client, waypoints[0], radius=0.10, rgba=(1.0, 0.8, 0.0, 1.0))
print(f"    [OK] {len(waypoints)} grey + {len(waypoints)} green (next {N_SHOW} visible) + 1 yellow\n")

def refresh_green_window(bc, adj_sph, adj_wps, cur_idx, n_show, buried):
    """Show green spheres only for the next n_show waypoints; bury all others."""
    for _i, _sid in enumerate(adj_sph):
        if cur_idx <= _i < cur_idx + n_show:
            move_sphere(bc, _sid, adj_wps[_i])
        else:
            move_sphere(bc, _sid, buried)

# Surface the first N_SHOW green spheres immediately
refresh_green_window(bc_client, adj_spheres, list(waypoints),
                     0, N_SHOW, _BURIED)

print("[4] Starting PRE-COMPENSATION EKF+LSTM mission...\n")

TAKEOFF_Z = 0.16
PATHFOLLOW_TRIGGER_Z = 0.15

env.set_target(np.array([0.0, 0.0, TAKEOFF_Z]))
path_active = False

dt = env.TIME_STEP

TARGET_SPEED = DEMO_TARGET_SPEED
total_distance = 0
for i in range(1, len(waypoints)):
    total_distance += np.linalg.norm(waypoints[i] - waypoints[i-1])

T_final = total_distance / TARGET_SPEED + 5.0
steps = int(T_final / dt)

PLAYBACK_SLOWDOWN = 1.0

print(f"Mission Duration: {T_final:.1f} seconds")
print(f"Search Area: 3.5m x 2.5m (scaled field)")
print(f"Total Distance: {total_distance:.1f}m at {TARGET_SPEED}m/s")
print(f"Playback Speed: {1/PLAYBACK_SLOWDOWN:.1f}x speed\n")

# GPS Dropout windows
num_dropouts = DEMO_NUM_DROPOUTS
dropout_windows = []
dropout_duration = DEMO_DROPOUT_DURATION_S

for i in range(num_dropouts):
    start_time = 3.0 + i * (T_final - 4.0) / max(num_dropouts, 1)
    end_time = start_time + dropout_duration
    if end_time < T_final - 1.0:
        dropout_windows.append((start_time, end_time))

print("GPS Dropout Windows:")
for i, (start, end) in enumerate(dropout_windows):
    print(f"  Window {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
print()

print("TIME   | GPS  | SOURCE           | EKF ERR | PRECOMP ERR | SHIFT | ALTITUDE | WPT")
print("-" * 92)

# State tracking
sensor_noise = EKFSensorNoise(
    sample_turn_on_bias_once=True,
    gyro_turn_on_bias_sigma=0.0,
)
sensor_noise.reset()
kf = QuadcopterEKF(
    dt=env.TIME_STEP,
    sigma_gyro_bias=EKF_SIGMA_GYRO_BIAS,
    q_bias_floor=EKF_Q_BIAS_FLOOR,
    initial_cov_diag=EKF_INITIAL_COV_DIAG.copy(),
)
kf.reset(
    state=np.concatenate(
        [
            obs[0:3].copy(),
            env.drone.xyz_dot.copy(),
            env.drone.rpy.copy(),
            env.drone.rpy_dot.copy(),
            true_gyro_bias_from_sensor_noise(sensor_noise),
        ]
    )
)
ekf_pos = kf.position.copy()
ekf_errors = []
precomp_errors = []   # error when using pre-compensated waypoints
prev_gps_ok = True
current_waypoint_idx = 0
dropout_duration_s = 0.0
dropout_anchor_pos = kf.position.copy()

# Accelerometer-derived velocity tracking for dropout
v_true_prev = np.asarray(env.drone.xyz_dot, dtype=float).copy()
v_acc_integrated = kf.velocity.copy()
ACCEL_VEL_BASE_STD = 0.05   # initial std on integrated-velocity measurement (m/s)
ACCEL_VEL_DRIFT_STD = 0.10  # additional std per second of dropout (m/s per s)

# Pre-compensation state
steps_since_dropout  = 0          # steps elapsed since GPS last recovered
adjusted_waypoints   = waypoints.copy()  # LSTM-adjusted targets (reset on GPS recovery)

waypoint_tolerance = 0.5
reached_waypoints = 0

# Unified IMU buffer: [vel(3), gyro(3), t_norm(1)] per step — 7 features, 300 steps
imu_buffer  = np.zeros((300, 7))
buffer_idx  = 0

# Position history for tracers
position_history = []  # list of (position, gps_status) tuples
last_tracer_step = 0
TRACER_UPDATE_INTERVAL = 10  # Draw tracer every 10 steps

# Logs for end-of-run comparison plots
log_t = []
log_true = []
log_ekf = []
log_precomp = []
log_ref = []
log_dropout = []
log_alt = []
log_shift = []

# =========================================================
# MAIN LOOP
# =========================================================
try:
    for k in range(steps):
        time.sleep(dt * PLAYBACK_SLOWDOWN)
        
        env.mission_time += dt
        t = env.mission_time

        # ---- GPS availability for this timestep ----
        gps_ok = not any(s <= t < e for s, e in dropout_windows)

        # ---- Sensor readings ----
        # true_pos / v_true / ang / rate are always read from physics.
        # During dropout, true_pos and v_true are NOT given to the controller.
        true_pos = env.drone.xyz
        v_true   = env.drone.xyz_dot
        ang      = env.drone.rpy
        rate     = env.drone.rpy_dot
        active_shift_mag = 0.0

        # ---- Step physics engine first so IMU obs are current ----
        # (We still need PID action from previous estimates — see below.)

        # ---- Liftoff detection via barometric altitude (always available) ----
        if not path_active and true_pos[2] > PATHFOLLOW_TRIGGER_Z:
            print(f"[*] Path following activated at t={t:.1f}s, z={true_pos[2]:.3f}m")
            path_active = True

        # ---- Position fed to the controller ----
        # GPS available  → real GPS position (all axes).
        # GPS dropout    → EKF dead-reckoning (XY); barometer (Z).
        #                  LSTM pre-compensation shifts waypoints, NOT EKF position.
        if gps_ok:
            pos_for_control = true_pos.copy()
            vel_for_control = v_true.copy()
        else:
            # XY: dropout EKF dead-reckoning; Z: barometric altitude (GPS-independent).
            pos_for_control = np.array([ekf_pos[0], ekf_pos[1], true_pos[2]], dtype=float)
            vel_for_control = kf.velocity.copy()

        # ---- Waypoint following — use pre-compensated targets during dropout ----
        if path_active and current_waypoint_idx < len(waypoints):
            target_wp  = adjusted_waypoints[current_waypoint_idx]
            dist_to_wp = np.linalg.norm(pos_for_control - target_wp)

            if dist_to_wp < waypoint_tolerance:
                # Dim the just-reached grey sphere; bury its green sphere
                wi_done = current_waypoint_idx
                recolor_sphere(bc_client, orig_spheres[wi_done], (0.9, 0.9, 0.9, 0.25))
                move_sphere(bc_client, adj_spheres[wi_done], _BURIED)
                reached_waypoints += 1
                current_waypoint_idx += 1
                if current_waypoint_idx < len(waypoints):
                    target_wp = adjusted_waypoints[current_waypoint_idx]
                    # Slide the visible green window forward and update yellow
                    refresh_green_window(bc_client, adj_spheres, adjusted_waypoints,
                                         current_waypoint_idx, N_SHOW, _BURIED)
                    move_sphere(bc_client, curr_sphere, adjusted_waypoints[current_waypoint_idx])
        else:
            target_wp = adjusted_waypoints[-1] if len(waypoints) > 0 else np.array([0, 0, 1.0])

        # ---- PID controller (no GPS position during dropout) ----
        quad.inject_external_state(pos_for_control, vel_for_control, ang, rate)
        z_ref = env.get_mission_reference()[2] if not path_active else target_wp[2]

        ctrl = quad.step(
            target_wp[:3] if path_active else env.get_mission_reference(),
            np.zeros(3),
            z_ref=z_ref,
            freeze_z_integrator=not gps_ok,
            control_profile="nominal" if gps_ok else "dropout",
        )
        rates_des = ctrl["rates_des"]
        U1 = ctrl["thrust_cmd"]

        # Build PID action
        pid_action = np.zeros(4, dtype=np.float32)
        pid_action[0] = thrust_to_action(U1, quad.m, quad.g)
        pid_action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

        # ---- Step physics engine ----
        obs, reward, done, truncated, info = env.step(pid_action)

        # ---- Tuned 15-state EKF path ----
        # World-frame acceleration from numerical differentiation of true velocity.
        # In a real system this would come from an IMU; here we synthesize it and
        # add the same noise model so the EKF treats it consistently.
        true_acc_world = (np.asarray(v_true, dtype=float).reshape(3) - v_true_prev) / dt
        v_true_prev = np.asarray(v_true, dtype=float).reshape(3).copy()

        noisy_pos, noisy_vel, noisy_att, noisy_rates, noisy_acc = sensor_noise.add_noise(
            pos=np.asarray(true_pos, dtype=float).reshape(3),
            vel=np.asarray(v_true, dtype=float).reshape(3),
            rot=np.asarray(ang, dtype=float).reshape(3),
            omega=np.asarray(rate, dtype=float).reshape(3),
            acc=true_acc_world,
            dt=dt,
        )
        imu_g = noisy_rates.copy()
        baro_z = sensor_noise.add_noise_to_baro(float(true_pos[2]))
        omega_predict = motor_omega_from_applied_forces(env, kf)

        if gps_ok:
            dropout_duration_s = 0.0
            kf.predict(omega=omega_predict, dt=dt)
            update_components = [
                kf.build_attitude_measurement(noisy_att),
                kf.build_gyro_measurement(noisy_rates),
                kf.build_velocity_measurement(noisy_vel),
                kf.build_baro_measurement(baro_z, std=sensor_noise.baro_noise_std),
                kf.build_gps_measurement(noisy_pos),
            ]
        else:
            # Snapshot velocity from EKF at dropout entry; integrate accelerometer
            # from this anchor so we have a velocity reference that reflects actual
            # motion, not a pseudo-zero assumption.
            if prev_gps_ok:
                v_acc_integrated = kf.velocity.copy()
            v_acc_integrated = v_acc_integrated + noisy_acc * dt

            dropout_duration_s += dt
            kf.predict_dropout(
                omega=omega_predict,
                dt=dt,
                dropout_time=dropout_duration_s,
            )
            # Velocity measurement built from accel integration (random-walk std
            # grows with dropout duration). This replaces the old pseudo-zero
            # update which dragged velocity toward zero while the drone was moving.
            accel_vel_std = ACCEL_VEL_BASE_STD + ACCEL_VEL_DRIFT_STD * dropout_duration_s
            update_components = [
                kf.build_attitude_measurement(noisy_att),
                kf.build_gyro_measurement(noisy_rates),
                kf.build_velocity_pseudo_measurement(
                    v_acc_integrated,
                    std_xy=accel_vel_std,
                    std_z=accel_vel_std * 1.5,
                ),
                kf.build_baro_measurement(baro_z, std=sensor_noise.baro_noise_std),
            ]

        for z_u, H_u, R_u in update_components:
            kf.update(measurement=z_u, H=H_u, measurement_noise=R_u)

        ekf_pos = kf.position.copy()

        # EKF position error vs ground truth (evaluation only)
        error_ekf = np.linalg.norm(ekf_pos - true_pos)
        ekf_errors.append(error_ekf)

        # ---- Track time since dropout for t_norm feature ----
        if not prev_gps_ok and gps_ok:
            # GPS just recovered: reset pre-compensation state
            steps_since_dropout = 0
            adjusted_waypoints  = waypoints.copy()
            # Snap visible green window back to original positions
            refresh_green_window(bc_client, adj_spheres, adjusted_waypoints,
                                 current_waypoint_idx, N_SHOW, _BURIED)
        elif prev_gps_ok and not gps_ok:
            dropout_anchor_pos = ekf_pos.copy()
        elif not gps_ok:
            steps_since_dropout += 1

        t_norm = min(steps_since_dropout / SHADOW_RESET_INTERVAL_PRECOMP, 1.0)

        # ---- Fill unified IMU buffer: [vel, gyro, t_norm] ----
        slot = buffer_idx % 300
        imu_buffer[slot] = np.concatenate([noisy_vel, imu_g, [t_norm]])
        buffer_idx += 1

        # ---- LSTM pre-compensation inference (only during GPS dropout) ----
        if buffer_idx >= 300 and not gps_ok and (steps_since_dropout * dt) >= PRECOMP_DELAY_S:
            # Roll circular buffer so oldest entry is first (matches training order)
            roll_by    = -(buffer_idx % 300)
            imu_ordered = np.roll(imu_buffer, roll_by, axis=0)
            imu_tensor  = torch.from_numpy(imu_ordered.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(imu_tensor)  # (1, 7, 3)

            dropout_elapsed_s = steps_since_dropout * dt
            ramp_alpha = float(
                np.clip(
                    (dropout_elapsed_s - PRECOMP_DELAY_S) / max(PRECOMP_RAMP_DURATION_S, 1.0e-6),
                    0.0,
                    1.0,
                )
            )
            effective_shift_cap = (
                PRECOMP_INITIAL_SHIFT_CAP
                + ramp_alpha * (MAX_PRECOMP_SHIFT - PRECOMP_INITIAL_SHIFT_CAP)
            )
            anchor_drift_xy = float(np.linalg.norm(ekf_pos[:2] - dropout_anchor_pos[:2]))
            drift_alpha = float(
                np.clip(
                    (anchor_drift_xy - PRECOMP_MIN_ANCHOR_DRIFT_M)
                    / max(PRECOMP_FULL_ANCHOR_DRIFT_M - PRECOMP_MIN_ANCHOR_DRIFT_M, 1.0e-6),
                    0.0,
                    1.0,
                )
            )
            effective_shift_cap *= drift_alpha

            # For each upcoming waypoint, estimate arrival time and select the
            # nearest LSTM horizon to predict drift at that moment.
            TARGET_SPEED = DEMO_TARGET_SPEED
            for wi in range(current_waypoint_idx, min(current_waypoint_idx + 5, len(waypoints))):
                dist = float(np.linalg.norm(ekf_pos - waypoints[wi]))
                t_arrival_s = dist / TARGET_SPEED if dist > 1e-3 else 0.0
                # Map arrival time to LSTM horizon index (clamp to 0..6)
                horizon_idx = min(
                    int(t_arrival_s / (HORIZON_STRIDE_PRECOMP * dt)),
                    NUM_HORIZONS_PRECOMP - 1
                )
                predicted_drift = preds[0, horizon_idx, :].cpu().numpy()
                # Pre-compensate XY only. Z is barometer-controlled (always accurate,
                # GPS-independent) so Z drift compensation is not only unnecessary
                # but actively harmful — it would command the drone to change altitude.
                predicted_drift[2] = 0.0
                approach_xy = waypoints[wi][:2] - ekf_pos[:2]
                a_norm = np.linalg.norm(approach_xy)
                if a_norm > 1.0e-3:
                    u = approach_xy / a_norm
                    shift_xy = np.dot(predicted_drift[:2], u) * u
                else:
                    shift_xy = predicted_drift[:2]
                shift_mag = np.linalg.norm(shift_xy)
                if shift_mag > effective_shift_cap:
                    shift_xy = shift_xy * (effective_shift_cap / max(shift_mag, 1.0e-9))
                # Safety: never shift past a fraction of the way to the waypoint,
                # otherwise we could plant the target behind the drone.
                wp_distance_xy = float(np.linalg.norm(waypoints[wi][:2] - ekf_pos[:2]))
                wp_shift_limit = PRECOMP_WAYPOINT_FRACTION * wp_distance_xy
                shift_mag = np.linalg.norm(shift_xy)
                if shift_mag > wp_shift_limit and wp_shift_limit > 0.0:
                    shift_xy = shift_xy * (wp_shift_limit / max(shift_mag, 1.0e-9))

                current_shift_xy = adjusted_waypoints[wi][:2] - waypoints[wi][:2]
                delta_shift_xy = shift_xy - current_shift_xy
                delta_mag = np.linalg.norm(delta_shift_xy)
                if delta_mag > MAX_SHIFT_DELTA_PER_UPDATE:
                    delta_shift_xy = delta_shift_xy * (
                        MAX_SHIFT_DELTA_PER_UPDATE / max(delta_mag, 1.0e-9)
                    )
                shift_xy = current_shift_xy + delta_shift_xy

                adjusted_waypoints[wi] = waypoints[wi] + np.array(
                    [shift_xy[0], shift_xy[1], 0.0],
                    dtype=float,
                )
                if wi == current_waypoint_idx:
                    active_shift_mag = float(np.linalg.norm(shift_xy))
            # Refresh the visible green window (next N_SHOW only) and yellow indicator
            refresh_green_window(bc_client, adj_spheres, adjusted_waypoints,
                                 current_waypoint_idx, N_SHOW, _BURIED)
            if current_waypoint_idx < len(waypoints):
                move_sphere(bc_client, curr_sphere, adjusted_waypoints[current_waypoint_idx])
        elif gps_ok:
            adjusted_waypoints = waypoints.copy()
            refresh_green_window(bc_client, adj_spheres, adjusted_waypoints,
                                 current_waypoint_idx, N_SHOW, _BURIED)
        elif current_waypoint_idx < len(waypoints):
            active_shift_mag = float(
                np.linalg.norm(adjusted_waypoints[current_waypoint_idx][:2] - waypoints[current_waypoint_idx][:2])
            )

        # ---- Pre-compensation position error (for display only) ----
        # Measures EKF error relative to the pre-compensated target frame
        if current_waypoint_idx < len(waypoints):
            offset = adjusted_waypoints[current_waypoint_idx] - waypoints[current_waypoint_idx]
            precomp_virtual_pos = ekf_pos - offset
            error_precomp = np.linalg.norm(precomp_virtual_pos - true_pos)
        else:
            precomp_virtual_pos = ekf_pos.copy()
            error_precomp = error_ekf
        precomp_errors.append(error_precomp)

        ref_pos = waypoints[min(current_waypoint_idx, len(waypoints) - 1)].copy()
        log_t.append(float(t))
        log_true.append(np.asarray(true_pos, dtype=float).copy())
        log_ekf.append(np.asarray(ekf_pos, dtype=float).copy())
        log_precomp.append(np.asarray(precomp_virtual_pos, dtype=float).copy())
        log_ref.append(np.asarray(ref_pos, dtype=float).copy())
        log_dropout.append(not gps_ok)
        log_alt.append(float(true_pos[2]))
        log_shift.append(float(active_shift_mag))

        # Store true position for visualisation tracers
        position_history.append((true_pos.copy(), gps_ok))

        # Draw tracers every TRACER_UPDATE_INTERVAL steps
        if k > TRACER_UPDATE_INTERVAL and (k - last_tracer_step) >= TRACER_UPDATE_INTERVAL:
            prev_idx = len(position_history) - TRACER_UPDATE_INTERVAL - 1
            curr_idx = len(position_history) - 1

            if prev_idx >= 0 and prev_idx < len(position_history):
                prev_pos, prev_gps = position_history[prev_idx]
                curr_pos, curr_gps = position_history[curr_idx]

                color = (0, 0, 1) if curr_gps else (1, 0, 0)
                draw_line(bc_client, prev_pos, curr_pos, color, width=2)

            last_tracer_step = k

        prev_gps_ok = gps_ok

        # Print progress
        if k % 50 == 0:
            gps_str    = "ON " if gps_ok else "OFF"
            source_str = "GPS      " if gps_ok else ("EKF+PRECOMP" if buffer_idx >= 300 else "EKF      ")
            wpt_str    = f"{current_waypoint_idx}/{len(waypoints)}"
            print(
                f"{t:6.1f}s| {gps_str} | {source_str} | "
                f"{error_ekf*100:6.2f}cm | {error_precomp*100:8.2f}cm | "
                f"{active_shift_mag*100:5.1f}cm | {true_pos[2]:7.3f}m | {wpt_str}"
            )

        if done or truncated:
            print(f"\n[!] Mission ended at step {k}")
            break

except KeyboardInterrupt:
    print("\n[!] Mission interrupted by user")

# =========================================================
# Results Summary
# =========================================================
print("\n" + "="*80)
print("MISSION COMPLETE - RESULTS")
print("="*80 + "\n")

print(f"Waypoints Reached: {reached_waypoints}/{len(waypoints)}")
print(f"Mission Duration: {env.mission_time:.1f} seconds")
print(f"Total Steps: {k+1}\n")

if len(ekf_errors) > 0:
    ekf_mean = np.mean(ekf_errors)
    ekf_std  = np.std(ekf_errors)
    pc_mean  = np.mean(precomp_errors)
    pc_std   = np.std(precomp_errors)
    shift_arr = np.asarray(log_shift, dtype=float) if log_shift else np.zeros(0, dtype=float)
    dropout_shift = shift_arr[np.asarray(log_dropout, dtype=bool)] if log_shift else np.zeros(0, dtype=float)

    print("Position Error Statistics:")
    print(f"  Pure EKF:              {ekf_mean*100:.2f} \u00b1 {ekf_std*100:.2f} cm")
    print(f"  EKF + Pre-Compensation:{pc_mean*100:.2f} \u00b1 {pc_std*100:.2f} cm")

    improvement = (pc_mean - ekf_mean) / ekf_mean * 100
    print(f"  Pre-Comp vs EKF:       {improvement:+.1f}%\n")
    if dropout_shift.size > 0:
        print("LSTM Shift Statistics (dropout only):")
        print(f"  Mean active shift:     {np.mean(dropout_shift)*100:.2f} cm")
        print(f"  Max active shift:      {np.max(dropout_shift)*100:.2f} cm\n")

_results_json = Path('experiments/lstm_position_drift_precomp_results.json')
if _results_json.exists():
    import json as _json
    with open(_results_json) as _f:
        _r = _json.load(_f)
    _missions  = _r.get('dataset', {}).get('missions', '?')
    _sequences = _r.get('dataset', {}).get('total_sequences', '?')
    _test_loss = _r.get('results', {}).get('test_loss', None)
    _training_info  = f"{_missions} missions, {_sequences:,} sequences" if isinstance(_sequences, int) else f"{_missions} missions"
    _test_loss_info = f"{_test_loss:.5f}" if _test_loss is not None else "N/A"
else:
    _training_info  = "unknown"
    _test_loss_info = "N/A"

print("Pre-Compensation Model Configuration:")
print(f"  Architecture: 256 hidden, 1 LSTM layer, 7 horizons (+0.5s..+3.5s)")
print(f"  Input: vel(3) + gyro(3) + t_norm(1) = 7 features")
print(f"  Training: {_training_info}")
print(f"  Test Loss: {_test_loss_info}\n")

print(f"Search Pattern: {pattern_type.upper()}")
print(f"Total Distance: {total_distance:.1f}m")
print(f"GPS Dropouts: {len(dropout_windows)} windows × {dropout_duration:.1f}s\n")

print("="*80)
print("[SUCCESS] Adaptive search mission with pre-compensation complete!")
print("="*80 + "\n")

if log_t:
    t_arr = np.asarray(log_t, dtype=float)
    true_arr = np.asarray(log_true, dtype=float)
    ekf_arr = np.asarray(log_ekf, dtype=float)
    precomp_arr = np.asarray(log_precomp, dtype=float)
    ref_arr = np.asarray(log_ref, dtype=float)
    dropout_mask = np.asarray(log_dropout, dtype=bool)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axis_labels = ["X [m]", "Y [m]", "Z [m]"]
    for i in range(3):
        for start_s, end_s in dropout_windows:
            axs[i].axvspan(start_s, end_s, color="0.9", alpha=0.5, zorder=0)
        axs[i].plot(t_arr, ref_arr[:, i], "--", color="black", linewidth=1.5, label="reference")
        axs[i].plot(t_arr, true_arr[:, i], color="0.4", linewidth=1.2, label="true")
        axs[i].plot(t_arr, ekf_arr[:, i], color="tab:blue", linewidth=1.5, label="EKF")
        axs[i].plot(t_arr, precomp_arr[:, i], color="tab:red", linewidth=1.5, label="LSTM-compensated")
        axs[i].set_ylabel(axis_labels[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Square Mission: Reference, Truth, EKF, and LSTM-Compensated Estimate")
    fig.tight_layout()

    ekf_err_arr = np.linalg.norm(ekf_arr - true_arr, axis=1)
    precomp_err_arr = np.linalg.norm(precomp_arr - true_arr, axis=1)
    fig_err, ax_err = plt.subplots(1, 1, figsize=(10, 4))
    for start_s, end_s in dropout_windows:
        ax_err.axvspan(start_s, end_s, color="0.9", alpha=0.5, zorder=0)
    ax_err.plot(t_arr, ekf_err_arr, color="tab:blue", linewidth=1.8, label="EKF error")
    ax_err.plot(t_arr, precomp_err_arr, color="tab:red", linewidth=1.8, label="LSTM-compensated error")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Position Error [m]")
    ax_err.set_title("EKF vs LSTM-Compensated Position Error")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")
    fig_err.tight_layout()

    shift_arr = np.asarray(log_shift, dtype=float)
    fig_shift, ax_shift = plt.subplots(1, 1, figsize=(10, 4))
    for start_s, end_s in dropout_windows:
        ax_shift.axvspan(start_s, end_s, color="0.9", alpha=0.5, zorder=0)
    ax_shift.plot(t_arr, shift_arr, color="tab:green", linewidth=1.8, label="active LSTM waypoint shift")
    ax_shift.axhline(MAX_PRECOMP_SHIFT, color="black", linestyle="--", linewidth=1.0, label="shift cap")
    ax_shift.set_xlabel("Time [s]")
    ax_shift.set_ylabel("Shift Magnitude [m]")
    ax_shift.set_title("LSTM Waypoint Shift During Dropout")
    ax_shift.grid(True, alpha=0.3)
    ax_shift.legend(loc="best")
    fig_shift.tight_layout()

env.close()
if log_t:
    plt.show()
