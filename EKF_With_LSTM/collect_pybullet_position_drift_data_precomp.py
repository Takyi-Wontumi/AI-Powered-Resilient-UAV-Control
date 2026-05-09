#!/usr/bin/env python3
"""
Collect Position-Drift Training Data from PyBullet
==================================================
Uses real PyBullet drone physics to generate training data for the position-drift LSTM.

WHY THIS IS NEEDED
------------------
The synthetic generator (generate_position_drift_data.py) approximates gyro as:
    gyro = cross(vel_normalized, pos_err_normalized) * 0.5
This produces values of ~0–0.3 rad/s.  A real quadrotor on a 3.5×2.5m perimeter
makes 90° turns that produce 5–15 rad/s.  The LSTM has never seen these signatures,
so it cannot predict drift during turns.

THIS COLLECTOR
--------------
  - Uses DroneMissionEnv + QuadcopterPID + the tuned 15-state QuadcopterEKF
  - Records env.drone.xyz_dot and obs[10:13] — the same signals stored in the
    demo's imu_buffer_a / imu_buffer_g at inference time
  - Runs circle / hover / square / figure-8 patterns for quick residual-learning data
  - Measures dropout EKF drift via a "shadow" 15-state EKF that never receives GPS

OUTPUT FORMAT
-------------
  Identical to generate_position_drift_data.py so train_lstm_position_drift.py
  can load it directly (saved as position_drift_pybullet_Nmissions_timestamp.pkl).

COMBINING WITH SYNTHETIC DATA
------------------------------
  train_lstm_position_drift.py is updated to load ALL position_drift_*.pkl files
  and concatenate them, so both datasets are used together.
"""

import sys, os
import argparse

# Force PyBullet into DIRECT (no-GUI) mode before any import
os.environ['PYBULLET_USE_SHARED_MEMORY'] = '0'

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for path in [
    SCRIPT_DIR,
    ROOT_DIR,
    os.path.join(ROOT_DIR, "AI_UAV_Tests"),
    os.path.join(ROOT_DIR, "GPS_Dropout_Recovery"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pickle
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect precomputed PyBullet position-drift missions for LSTM training."
    )
    p.add_argument(
        "--patterns",
        type=str,
        default=None,
        help=(
            "Comma-separated pattern list to sample from, for example "
            "'square,hover,circle,helix'. Overrides --pattern."
        ),
    )
    p.add_argument(
        "--pattern",
        choices=["square", "circle", "hover", "figure8", "helix", "mixed"],
        default="square",
        help=(
            "Mission pattern to collect. 'mixed' samples across the recommended "
            "training mix: square, hover, circle, helix."
        ),
    )
    p.add_argument(
        "--num-missions",
        type=int,
        default=600,
        help="Number of missions to collect.",
    )
    p.add_argument(
        "--steps-per-mission",
        type=int,
        default=6000,
        help="Simulation steps per mission.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="data/position_drift",
        help="Directory where the dataset pickle will be written.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for repeatable quick runs.",
    )
    p.add_argument(
        "--square-points-per-edge",
        type=int,
        default=6,
        help="Waypoint density for square missions.",
    )
    return p.parse_args()


args = parse_args()
np.random.seed(args.seed)

print("\n" + "="*100)
print("COLLECT POSITION-DRIFT TRAINING DATA FROM PYBULLET")
print("="*100 + "\n")

# =========================================================
# Configuration
# =========================================================
NUM_MISSIONS     = int(args.num_missions)   # target missions — high-volume run
STEPS_PER_MISSION = int(args.steps_per_mission)  # 30s at 200 Hz — needed for 2500-step reset windows
SHADOW_RESET_INTERVAL = 2500  # re-anchor shadow KF every 12.5s — covers 10s+ dropouts
SAVE_DIR         = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PSEUDO_VEL_STD_XY = 2.10
PSEUDO_VEL_STD_Z = 4.00
EKF_SIGMA_GYRO_BIAS = 0.001
EKF_Q_BIAS_FLOOR = 1.0e-8
EKF_INITIAL_GYRO_BIAS_STD = 0.001
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

SUPPORTED_PATTERNS = {"square", "circle", "hover", "figure8", "helix"}


def resolve_pattern_mix():
    if args.patterns is not None:
        patterns = [p.strip().lower() for p in str(args.patterns).split(",") if p.strip()]
        invalid = [p for p in patterns if p not in SUPPORTED_PATTERNS]
        if invalid:
            raise ValueError(
                f"Unsupported pattern(s) in --patterns: {invalid}. "
                f"Supported: {sorted(SUPPORTED_PATTERNS)}"
            )
        if not patterns:
            raise ValueError("No valid patterns were provided to --patterns.")
        return patterns, [1.0 / len(patterns)] * len(patterns)

    if args.pattern == "mixed":
        patterns = ["square", "hover", "circle", "helix"]
        return patterns, [0.25, 0.25, 0.25, 0.25]

    return [str(args.pattern)], [1.0]


PATTERNS, PATTERN_WEIGHTS = resolve_pattern_mix()

# =========================================================
# Imports
# =========================================================
print("[1/4] Loading dependencies...")
try:
    from phoenix_drone_simulation.envs.control import AttitudeRate
    from phoenix_drone_simulation.envs.mission import DroneMissionEnv
    from quadcopter_env import QuadcopterPID
    from AI_UAV_Tests.quadcopter_ekf import QuadcopterEKF
    from AI_UAV_Tests.sensors_ekf import EKFSensorNoise
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Ensure phoenix_drone_simulation, quadcopter_env, quadcopter_ekf, sensors_ekf are on sys.path")
    sys.exit(1)
print("    [OK]\n")


# =========================================================
# Search-pattern generator  (matches demo exactly)
# =========================================================
class AdaptiveSearchPatternGenerator:
    """Generates compact training trajectories on a 3.5×2.5m field."""

    def __init__(self, area_size=(3.5, 2.5), altitude=1.0):
        self.x_min = -area_size[0] / 2
        self.x_max =  area_size[0] / 2
        self.y_min = -area_size[1] / 2
        self.y_max =  area_size[1] / 2
        self.z     = altitude
        self.area_size = area_size

    def circle_search(self, num_points=36, radius_scale=0.45, randomness=0.04):
        """Circle trajectory centered in the workspace with random chirality."""
        wps = []
        radius = min(self.x_max - self.x_min, self.y_max - self.y_min) * radius_scale
        chirality = 1.0 if np.random.random() < 0.5 else -1.0
        cx = np.random.uniform(self.x_min * 0.15, self.x_max * 0.15)
        cy = np.random.uniform(self.y_min * 0.15, self.y_max * 0.15)
        for i in range(num_points):
            theta = chirality * 2.0 * np.pi * i / num_points
            x = cx + radius * np.cos(theta) + np.random.randn() * randomness
            y = cy + radius * np.sin(theta) + np.random.randn() * randomness
            wps.append(np.array([
                np.clip(x, self.x_min, self.x_max),
                np.clip(y, self.y_min, self.y_max),
                self.z,
            ]))
        return np.array(wps)

    def square_search(self, side_scale=0.55, points_per_edge=8, randomness=0.03):
        """Square perimeter with random chirality and small waypoint jitter."""
        wps = []
        side_x = (self.x_max - self.x_min) * side_scale
        side_y = (self.y_max - self.y_min) * side_scale
        cx = np.random.uniform(self.x_min * 0.15, self.x_max * 0.15)
        cy = np.random.uniform(self.y_min * 0.15, self.y_max * 0.15)
        corners = np.array([
            [cx - side_x / 2.0, cy - side_y / 2.0, self.z],
            [cx + side_x / 2.0, cy - side_y / 2.0, self.z],
            [cx + side_x / 2.0, cy + side_y / 2.0, self.z],
            [cx - side_x / 2.0, cy + side_y / 2.0, self.z],
        ])
        if np.random.random() < 0.5:
            corners = corners[::-1]
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            for j in range(points_per_edge):
                alpha = j / points_per_edge
                pt = (1.0 - alpha) * start + alpha * end
                pt[0] += np.random.randn() * randomness
                pt[1] += np.random.randn() * randomness
                pt[0] = np.clip(pt[0], self.x_min, self.x_max)
                pt[1] = np.clip(pt[1], self.y_min, self.y_max)
                wps.append(pt.copy())
        return np.array(wps)

    def figure8_search(self, num_points=48, radius_scale=0.32, randomness=0.03):
        """Planar figure-8 using a lemniscate-like path."""
        wps = []
        chirality = 1.0 if np.random.random() < 0.5 else -1.0
        ax = (self.x_max - self.x_min) * radius_scale
        ay = (self.y_max - self.y_min) * radius_scale
        cx = np.random.uniform(self.x_min * 0.10, self.x_max * 0.10)
        cy = np.random.uniform(self.y_min * 0.10, self.y_max * 0.10)
        for i in range(num_points):
            t = chirality * 2.0 * np.pi * i / num_points
            x = cx + ax * np.sin(t)
            y = cy + ay * np.sin(t) * np.cos(t)
            x += np.random.randn() * randomness
            y += np.random.randn() * randomness
            wps.append(np.array([
                np.clip(x, self.x_min, self.x_max),
                np.clip(y, self.y_min, self.y_max),
                self.z,
            ]))
        return np.array(wps)

    def helix_search(
        self,
        num_points=48,
        radius_scale=0.30,
        z_amplitude=0.22,
        turns=1.25,
        randomness=0.02,
    ):
        """Bounded 3D helix for coupled lateral/vertical dropout drift data."""
        wps = []
        chirality = 1.0 if np.random.random() < 0.5 else -1.0
        radius = min(self.x_max - self.x_min, self.y_max - self.y_min) * radius_scale
        cx = np.random.uniform(self.x_min * 0.10, self.x_max * 0.10)
        cy = np.random.uniform(self.y_min * 0.10, self.y_max * 0.10)
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        for i in range(num_points):
            theta = chirality * turns * 2.0 * np.pi * i / max(1, num_points - 1)
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = self.z + z_amplitude * np.sin(theta + phase)
            x += np.random.randn() * randomness
            y += np.random.randn() * randomness
            z += np.random.randn() * (0.5 * randomness)
            wps.append(np.array([
                np.clip(x, self.x_min, self.x_max),
                np.clip(y, self.y_min, self.y_max),
                np.clip(z, 0.75, 1.25),
            ]))
        return np.array(wps)

    def hover_search(self):
        """Drone moves to a random point then holds still.
        Teaches the model that zero velocity + zero gyro = zero drift.
        Uses a few random intermediate waypoints so the approach varies."""
        wps = []
        # Move to 1-3 random waypoints first (so the buffer isn't all zeros)
        n_pre = np.random.randint(1, 4)
        for _ in range(n_pre):
            x = np.random.uniform(self.x_min * 0.7, self.x_max * 0.7)
            y = np.random.uniform(self.y_min * 0.7, self.y_max * 0.7)
            wps.append(np.array([x, y, self.z]))
        # Then a single hover point repeated many times (~hold for rest of mission)
        hx = np.random.uniform(self.x_min * 0.5, self.x_max * 0.5)
        hy = np.random.uniform(self.y_min * 0.5, self.y_max * 0.5)
        for _ in range(30):
            wps.append(np.array([hx, hy, self.z]))
        return np.array(wps)


def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    return float(np.clip((U1 / hover_T - 0.9) / 0.4, -1.0, 1.0))


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


# =========================================================
# Create ONE PyBullet env (reused across all missions via reset)
# =========================================================
print("[2/4] Creating PyBullet environment (headless)...")
try:
    env = DroneMissionEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode=None,     # headless — no GUI window
    )
    env.drone.control = AttitudeRate(
        bc=env.bc, drone=env.drone, time_step=env.TIME_STEP
    )
    obs, info = env.reset()
except Exception as e:
    print(f"  ERROR creating environment: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print(f"    [OK]  TIME_STEP={env.TIME_STEP:.4f}s  obs_shape={obs.shape}\n")

generator = AdaptiveSearchPatternGenerator(area_size=(3.5, 2.5), altitude=1.0)


# =========================================================
# Collection loop
# =========================================================
print(f"[3/4] Collecting {NUM_MISSIONS} PyBullet missions...\n")
print(f"  {'#':>4}  {'pattern':>10}  {'steps':>6}  {'drift_max':>10}  {'cumulative':>12}")
print("  " + "-"*50)

missions       = []
pattern_counts = {p: 0 for p in PATTERNS}
skipped        = 0

for mission_idx in range(NUM_MISSIONS):
    pattern = np.random.choice(PATTERNS, p=PATTERN_WEIGHTS)
    pattern_counts[pattern] += 1

    # ---- Generate waypoints ----
    try:
        if pattern == 'circle':
            wps = generator.circle_search(
                num_points=np.random.randint(28, 44),
                radius_scale=np.random.uniform(0.35, 0.48),
                randomness=np.random.uniform(0.01, 0.05))
        elif pattern == 'square':
            wps = generator.square_search(
                side_scale=np.random.uniform(0.45, 0.65),
                points_per_edge=max(3, int(args.square_points_per_edge)),
                randomness=np.random.uniform(0.01, 0.04))
        elif pattern == 'figure8':
            wps = generator.figure8_search(
                num_points=np.random.randint(36, 56),
                radius_scale=np.random.uniform(0.25, 0.36),
                randomness=np.random.uniform(0.01, 0.04))
        elif pattern == 'helix':
            wps = generator.helix_search(
                num_points=np.random.randint(40, 60),
                radius_scale=np.random.uniform(0.24, 0.34),
                z_amplitude=np.random.uniform(0.14, 0.26),
                turns=np.random.uniform(1.0, 1.5),
                randomness=np.random.uniform(0.01, 0.03))
        else:  # hover
            wps = generator.hover_search()
    except Exception as e:
        print(f"  [!] Waypoint generation failed (mission {mission_idx}): {e}")
        skipped += 1
        continue

    # ---- Reset environment ----
    try:
        obs, info = env.reset()
        env.drone.control = AttitudeRate(
            bc=env.bc, drone=env.drone, time_step=env.TIME_STEP
        )
    except Exception as e:
        print(f"  [!] env.reset() failed (mission {mission_idx}): {e}")
        skipped += 1
        continue

    # ---- Initialise controllers and filters ----
    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    sensor_noise = EKFSensorNoise(
        sample_turn_on_bias_once=True,
        gyro_turn_on_bias_sigma=0.0,
    )
    pos0 = obs[0:3].copy()
    vel0 = env.drone.xyz_dot.copy()
    att0 = env.drone.rpy.copy()
    rate0 = env.drone.rpy_dot.copy()

    sensor_noise.reset()
    kf_shadow = QuadcopterEKF(
        dt=env.TIME_STEP,
        sigma_gyro_bias=EKF_SIGMA_GYRO_BIAS,
        q_bias_floor=EKF_Q_BIAS_FLOOR,
        initial_cov_diag=EKF_INITIAL_COV_DIAG.copy(),
    )
    kf_shadow.reset(
        state=np.concatenate(
            [
                pos0,
                vel0,
                att0,
                rate0,
                true_gyro_bias_from_sensor_noise(sensor_noise),
            ]
        )
    )

    # ---- Storage ----
    vel_list    = []
    gyro_list   = []
    true_list   = []
    dr_list     = []
    t_norm_list = []   # time since last shadow-KF reset, normalised 0->1

    wp_idx      = 0
    path_active = False
    TRIGGER_Z   = 0.15
    WP_TOL      = 0.5

    done = False

    # Accelerometer-derived velocity tracking — must mirror demo behavior so the
    # LSTM is trained on the same EKF dynamics it will see at deployment.
    v_true_prev = vel0.copy()
    v_acc_integrated = vel0.copy()
    ACCEL_VEL_BASE_STD = 0.05
    ACCEL_VEL_DRIFT_STD = 0.10

    for step in range(STEPS_PER_MISSION):
        # ---- Read physics state ----
        true_pos = env.drone.xyz       # world-frame position (ground truth)
        v_true   = env.drone.xyz_dot   # world-frame velocity
        ang      = env.drone.rpy       # roll/pitch/yaw
        rate     = env.drone.rpy_dot   # angular rate (for PID)

        # ---- Takeoff detection ----
        if not path_active and true_pos[2] > TRIGGER_Z:
            path_active = True

        # ---- Waypoint advance ----
        if path_active and wp_idx < len(wps):
            if np.linalg.norm(true_pos - wps[wp_idx]) < WP_TOL:
                wp_idx = (wp_idx + 1) % len(wps)

        target_wp = wps[wp_idx % len(wps)]

        # ---- PID control (always with true GPS — only for flying, not collected) ----
        quad.inject_external_state(true_pos, v_true, ang, rate)
        z_ref = target_wp[2] if path_active else 1.0
        ref   = target_wp if path_active else np.array([0.0, 0.0, 1.0])
        ctrl  = quad.step(ref, np.zeros(3), z_ref=z_ref)

        pid_action    = np.zeros(4, dtype=np.float32)
        pid_action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        pid_action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        # ---- Step physics ----
        try:
            obs, _, done, truncated, _ = env.step(pid_action)
        except Exception:
            done = True

        # ---- Shadow EKF: reset to truth every window, then run dropout-only updates ----
        # This simulates repeated short GPS dropout windows using the tuned 15-state EKF.
        if step > 0 and step % SHADOW_RESET_INTERVAL == 0:
            kf_shadow.reset(
                state=np.concatenate(
                    [
                        obs[0:3].copy(),
                        v_true.copy(),
                        ang.copy(),
                        rate.copy(),
                        kf_shadow.gyro_bias.copy(),
                    ]
                )
            )
            # Re-anchor accel-integrated velocity at each shadow reset window
            v_acc_integrated = v_true.copy()

        # World-frame acceleration from numerical differentiation, fed into the
        # noise model so the EKF can use a noisy accel reading during dropout.
        true_acc_world = (np.asarray(v_true, dtype=float).reshape(3) - v_true_prev) / env.TIME_STEP
        v_true_prev = np.asarray(v_true, dtype=float).reshape(3).copy()

        noisy_pos, noisy_vel, noisy_att, noisy_rates, noisy_acc = sensor_noise.add_noise(
            pos=np.asarray(true_pos, dtype=float).reshape(3),
            vel=np.asarray(v_true, dtype=float).reshape(3),
            rot=np.asarray(ang, dtype=float).reshape(3),
            omega=np.asarray(rate, dtype=float).reshape(3),
            acc=true_acc_world,
            dt=env.TIME_STEP,
        )

        # Integrate accel into velocity estimate for the dropout-style update
        v_acc_integrated = v_acc_integrated + noisy_acc * env.TIME_STEP
        baro_z = sensor_noise.add_noise_to_baro(float(true_pos[2]))
        imu_g = noisy_rates.copy()

        omega_predict = motor_omega_from_applied_forces(env, kf_shadow)
        dropout_time = (step % SHADOW_RESET_INTERVAL) * env.TIME_STEP
        kf_shadow.predict_dropout(
            omega=omega_predict,
            dt=env.TIME_STEP,
            dropout_time=dropout_time,
        )

        # Velocity measurement uses accel-integrated estimate (snapshotted at
        # each shadow reset). Std grows linearly with time-since-reset to model
        # accumulating integration noise.
        accel_vel_std = ACCEL_VEL_BASE_STD + ACCEL_VEL_DRIFT_STD * dropout_time
        update_components = [
            kf_shadow.build_attitude_measurement(noisy_att),
            kf_shadow.build_gyro_measurement(noisy_rates),
            kf_shadow.build_velocity_pseudo_measurement(
                v_acc_integrated,
                std_xy=accel_vel_std,
                std_z=accel_vel_std * 1.5,
            ),
            kf_shadow.build_baro_measurement(
                baro_z,
                std=sensor_noise.baro_noise_std,
            ),
        ]
        for z_u, H_u, R_u in update_components:
            kf_shadow.update(measurement=z_u, H=H_u, measurement_noise=R_u)

        # Compute time-since-reset feature (normalised 0->1 over one reset window).
        # This tells the model how far into a dropout window it currently is.
        steps_since_reset = step % SHADOW_RESET_INTERVAL
        t_norm = steps_since_reset / SHADOW_RESET_INTERVAL

        # ---- Record inputs and EKF-position drift target ----
        vel_list.append(noisy_vel.copy())
        gyro_list.append(imu_g.copy())
        true_list.append(obs[0:3].copy())
        dr_list.append(kf_shadow.position.copy())
        t_norm_list.append(np.float32(t_norm))

        if done or truncated:
            break

    # ---- Discard missions that crashed early ----
    n = len(vel_list)
    if n < 500:
        print(f"  [!] Mission {mission_idx:3d}: only {n} steps — skipping")
        skipped += 1
        continue

    drift_arr = np.array(dr_list, dtype=np.float32) - np.array(true_list, dtype=np.float32)
    max_drift = float(np.max(np.linalg.norm(drift_arr, axis=1)))

    missions.append({
        'vel_meas':       np.array(vel_list,   dtype=np.float32),   # (T, 3)
        'gyro_meas':      np.array(gyro_list,  dtype=np.float32),   # (T, 3)
        'true_pos':       np.array(true_list,  dtype=np.float32),   # (T, 3)
        'dr_pos':         np.array(dr_list,    dtype=np.float32),   # (T, 3) shadow 15-state EKF position
        't_norm_meas':    np.array(t_norm_list, dtype=np.float32),  # (T,)  time-since-reset 0->1
        'mission_type':   pattern,
        'duration_steps': n,
        'source':         'pybullet_precomp',
        'estimator':      'quadcopter_ekf_15state',
    })

    cum = sum(m['duration_steps'] for m in missions)
    print(f"  {mission_idx+1:4d}  {pattern:>10}  {n:6d}  {max_drift:9.3f}m  {cum:12,}")


# =========================================================
# Save
# =========================================================
print(f"\n[4/4] Saving dataset...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path  = SAVE_DIR / f'position_drift_precomp_{len(missions)}missions_{timestamp}.pkl'

with open(out_path, 'wb') as f:
    pickle.dump(missions, f)

total_steps = sum(m['duration_steps'] for m in missions)

print(f"\n{'='*70}")
print(f"COLLECTION COMPLETE")
print(f"{'='*70}")
print(f"  Missions collected : {len(missions)}  (skipped: {skipped})")
print(f"  Total steps        : {total_steps:,}")
print(f"  Pattern mix        : {pattern_counts}")
print(f"  Saved to           : {out_path.name}")
print(f"\nNext: run train_lstm_position_drift_precomp.py to train on this tuned 15-state EKF dataset.")
