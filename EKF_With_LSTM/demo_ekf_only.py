#!/usr/bin/env python3
"""
Pure EKF Mission Demo
======================
Baseline comparison: drone navigates using raw EKF dead-reckoning only.
No LSTM pre-compensation, no RL correction.

During GPS dropout the drone follows EKF-estimated position — drift accumulates
uncorrected. Use this as the visual/numerical baseline against demo_rl_precomp.py.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'AI_UAV_Tests'))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GPS_Dropout_Recovery'))

import time
import numpy as np
from pathlib import Path

print("\n" + "=" * 80)
print("ADAPTIVE SEARCH MISSION — PURE EKF BASELINE DEMO")
print("=" * 80)
print("\nEKF dead-reckoning only — no LSTM, no RL correction\n")

# ── Constants ─────────────────────────────────────────────────────────────────
WAYPOINT_TOLERANCE   = 0.5
TARGET_SPEED         = 1.2
TAKEOFF_Z            = 0.16
PATHFOLLOW_TRIGGER_Z = 0.15
PLAYBACK_SLOWDOWN    = 1.0

# ── Environment ───────────────────────────────────────────────────────────────
print("[1] Creating PyBullet environment...")
from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from quadcopter_env import QuadcopterPID
from kalman_filter_ins import KalmanFilterINS

def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    return float(np.clip((U1 / hover_T - 0.9) / 0.4, -1.0, 1.0))

env = DroneMissionEnv(
    physics="PyBulletPhysics",
    control_mode="AttitudeRate",
    drone_model="cf21x_bullet",
    dropout_mode="NONE",
    render_mode="human",
)
env.drone.control = AttitudeRate(bc=env.bc, drone=env.drone, time_step=env.TIME_STEP)
obs, info = env.reset()
print("    [OK] Environment ready\n")

bc_client = env.bc

# ── Helper ────────────────────────────────────────────────────────────────────
def draw_line(bc, start, end, color, width=2):
    bc.addUserDebugLine(lineFromXYZ=start, lineToXYZ=end,
                        lineColorRGB=color, lineWidth=width)

# ── Mission generation (identical to demo_rl_precomp.py) ─────────────────────
print("[2] Generating randomised search mission...\n")

class AdaptiveSearchPatternGenerator:
    def __init__(self, area_size=(3.5, 2.5), altitude=1.0):
        self.x_min, self.x_max = -area_size[0] / 2, area_size[0] / 2
        self.y_min, self.y_max = -area_size[1] / 2, area_size[1] / 2
        self.z = altitude
        self.area_size = area_size

    def zigzag_search(self, num_passes=8, randomness=0.3):
        waypoints = []
        x_width   = self.x_max - self.x_min
        y_spacing = (self.y_max - self.y_min) / (num_passes + 1)
        for i in range(num_passes):
            y = self.y_min + (i + 1) * y_spacing
            x_start = np.clip(self.x_min + np.random.uniform(-randomness, randomness) * x_width * 0.05,
                              self.x_min, self.x_max)
            x_end   = np.clip(self.x_max + np.random.uniform(-randomness, randomness) * x_width * 0.05,
                              self.x_min, self.x_max)
            for j in range(3):
                x   = x_start + (x_end - x_start) * j / 2
                y_v = np.clip(y + np.random.uniform(-randomness, randomness) * y_spacing * 0.3,
                              self.y_min, self.y_max)
                waypoints.append(np.array([x, y_v, self.z]))
            if i < num_passes - 1:
                y_turn = np.clip(y + y_spacing * 0.5, self.y_min, self.y_max)
                waypoints.append(np.array([x_end, y_turn, self.z]))
        return np.array(waypoints)

    def spiral_search(self, rotations=4, randomness=0.2):
        waypoints = []
        n = 40
        for i in range(n):
            t = 2 * np.pi * rotations * i / n
            r = (self.area_size[0] / 2) * i / n
            x = np.clip(r * np.cos(t) + np.random.uniform(-randomness, randomness) * self.area_size[0] * 0.1,
                        self.x_min, self.x_max)
            y = np.clip((r / 1.5) * np.sin(t) + np.random.uniform(-randomness, randomness) * self.area_size[1] * 0.1,
                        self.y_min, self.y_max)
            waypoints.append(np.array([x, y, self.z]))
        return np.array(waypoints)

    def perimeter_search(self, num_laps=3, randomness=0.15):
        waypoints = []
        perimeter = 2 * (self.x_max - self.x_min) + 2 * (self.y_max - self.y_min)
        pts_per_lap = max(8, int(round(perimeter * 2.4)))
        for i in range(pts_per_lap * num_laps):
            frac = (i % pts_per_lap) / pts_per_lap
            p    = frac * perimeter
            if p < (self.x_max - self.x_min):
                x, y = self.x_min + p, self.y_min
            elif p < 2 * (self.x_max - self.x_min):
                x, y = self.x_max, self.y_min + (p - (self.x_max - self.x_min))
            elif p < 2 * (self.x_max - self.x_min) + (self.y_max - self.y_min):
                x, y = self.x_max - (p - 2 * (self.x_max - self.x_min)), self.y_max
            else:
                x, y = self.x_min, self.y_max - (p - 2 * (self.x_max - self.x_min) - (self.y_max - self.y_min))
            x = np.clip(x + np.random.uniform(-randomness, randomness) * (self.x_max - self.x_min) * 0.1,
                        self.x_min, self.x_max)
            y = np.clip(y + np.random.uniform(-randomness, randomness) * (self.y_max - self.y_min) * 0.1,
                        self.y_min, self.y_max)
            waypoints.append(np.array([x, y, self.z]))
        return np.array(waypoints)

generator    = AdaptiveSearchPatternGenerator(area_size=(3.5, 2.5), altitude=1.0)
pattern_type = np.random.choice(['zigzag', 'spiral', 'perimeter'])
print(f"Generated pattern: {pattern_type.upper()}")

if pattern_type == 'zigzag':
    waypoints = generator.zigzag_search()
elif pattern_type == 'spiral':
    waypoints = generator.spiral_search()
else:
    waypoints = generator.perimeter_search()

print(f"Generated {len(waypoints)} waypoints\n")

# ── Mission configuration ─────────────────────────────────────────────────────
quad = QuadcopterPID(dt=env.TIME_STEP)
quad.reset()
env.set_target(np.array([0.0, 0.0, TAKEOFF_Z]))

dt           = env.TIME_STEP
path_active  = False
total_dist   = sum(np.linalg.norm(waypoints[i] - waypoints[i-1]) for i in range(1, len(waypoints)))
T_final      = total_dist / TARGET_SPEED + 5.0
steps        = int(T_final / dt)

num_dropouts    = int(T_final / 10)
dropout_windows = []
for i in range(num_dropouts):
    start = 3.0 + i * (T_final - 4.0) / max(num_dropouts, 1)
    end   = start + 3.5
    if end < T_final - 1.0:
        dropout_windows.append((start, end))

print("[3] Starting pure EKF mission...\n")
print(f"Mission Duration: {T_final:.1f}s  |  Distance: {total_dist:.1f}m at {TARGET_SPEED}m/s")
print("GPS Dropout Windows:")
for i, (s, e) in enumerate(dropout_windows):
    print(f"  Window {i+1}: {s:.1f}s – {e:.1f}s  ({e-s:.1f}s)")
print()

print("TIME   | GPS  | EKF ERR  | TRUE POS XY           | EKF POS XY            | ALT    | WPT")
print("-" * 97)

# ── State ─────────────────────────────────────────────────────────────────────
kf      = KalmanFilterINS(dt=env.TIME_STEP)
kf.set_state(obs[0:3], env.drone.xyz_dot, env.drone.rpy)
ekf_pos = obs[0:3].copy()

ekf_errors          = []
prev_gps_ok         = True
current_waypoint_idx = 0
reached_waypoints   = 0
position_history    = []
last_tracer_step    = 0
TRACER_INTERVAL     = 10

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    for k in range(steps):
        time.sleep(dt * PLAYBACK_SLOWDOWN)

        env.mission_time += dt
        t = env.mission_time

        gps_ok   = not any(s <= t < e for s, e in dropout_windows)
        true_pos = env.drone.xyz.copy()
        v_true   = env.drone.xyz_dot.copy()
        ang      = env.drone.rpy.copy()
        rate     = env.drone.rpy_dot.copy()

        if not path_active and true_pos[2] > PATHFOLLOW_TRIGGER_Z:
            print(f"[*] Path following activated at t={t:.1f}s, z={true_pos[2]:.3f}m")
            path_active = True

        # Position source: GPS when available, EKF XY + baro Z during dropout
        if gps_ok:
            pos_for_control = true_pos.copy()
            vel_for_control = v_true.copy()
        else:
            pos_for_control = np.array([ekf_pos[0], ekf_pos[1], true_pos[2]])
            vel_for_control = v_true.copy()

        # Waypoint following — raw (uncompensated) waypoints only
        if path_active and current_waypoint_idx < len(waypoints):
            target_wp = waypoints[current_waypoint_idx]
            if np.linalg.norm(pos_for_control - target_wp) < WAYPOINT_TOLERANCE:
                reached_waypoints    += 1
                current_waypoint_idx += 1
                if current_waypoint_idx < len(waypoints):
                    target_wp = waypoints[current_waypoint_idx]
        else:
            target_wp = waypoints[-1] if len(waypoints) > 0 else np.array([0, 0, 1.0])

        # PID
        quad.inject_external_state(pos_for_control, vel_for_control, ang, rate)
        z_ref  = env.get_mission_reference()[2] if not path_active else target_wp[2]
        ctrl   = quad.step(target_wp[:3] if path_active else env.get_mission_reference(),
                           np.zeros(3), z_ref=z_ref)
        pid_action    = np.zeros(4, dtype=np.float32)
        pid_action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        pid_action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        obs, _, done, truncated, _ = env.step(pid_action)

        # EKF update
        kf.x[3:6] = v_true
        kf.x[6:9] = ang
        kf.predict()
        if gps_ok:
            kf.update_with_gps(obs[0:3])
        ekf_pos = kf.get_position().copy()

        error_ekf = float(np.linalg.norm(ekf_pos - true_pos))
        ekf_errors.append(error_ekf)

        position_history.append((true_pos.copy(), gps_ok))

        # Tracers — blue during GPS-on, red during dropout
        if k > TRACER_INTERVAL and (k - last_tracer_step) >= TRACER_INTERVAL:
            prev_idx = len(position_history) - TRACER_INTERVAL - 1
            if 0 <= prev_idx < len(position_history):
                prev_pos, prev_gps = position_history[prev_idx]
                curr_pos, curr_gps = position_history[-1]
                color = (0, 0.6, 1) if curr_gps else (1, 0.2, 0)
                draw_line(bc_client, prev_pos, curr_pos, color, width=2)
            last_tracer_step = k

        prev_gps_ok = gps_ok

        if k % 50 == 0:
            gps_str = "ON " if gps_ok else "OFF"
            src_str = "GPS" if gps_ok else "EKF"
            wpt_str = f"{current_waypoint_idx}/{len(waypoints)}"
            print(f"{t:6.1f}s| {gps_str} | {error_ekf*100:6.2f}cm"
                  f" | ({true_pos[0]:+6.3f}, {true_pos[1]:+6.3f})"
                  f" | ({ekf_pos[0]:+6.3f}, {ekf_pos[1]:+6.3f})"
                  f" | {true_pos[2]:5.3f}m | {wpt_str}")

        if done or truncated:
            print(f"\n[!] Mission ended at step {k}")
            break

except KeyboardInterrupt:
    print("\n[!] Mission interrupted by user")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("MISSION COMPLETE — RESULTS")
print("=" * 80 + "\n")

print(f"Waypoints Reached : {reached_waypoints}/{len(waypoints)}")
print(f"Mission Duration  : {env.mission_time:.1f}s")
print(f"Search Pattern    : {pattern_type.upper()}")
print(f"GPS Dropouts      : {len(dropout_windows)} window(s)\n")

if ekf_errors:
    ekf_mean = np.mean(ekf_errors)
    ekf_std  = np.std(ekf_errors)
    ekf_max  = np.max(ekf_errors)
    print("Position Error Statistics (Pure EKF — no compensation):")
    print(f"  Mean  : {ekf_mean*100:.2f} cm")
    print(f"  Std   : {ekf_std*100:.2f} cm")
    print(f"  Peak  : {ekf_max*100:.2f} cm")
    print()

print("=" * 80)
print("[DONE] Pure EKF baseline demo complete.")
print("=" * 80)
