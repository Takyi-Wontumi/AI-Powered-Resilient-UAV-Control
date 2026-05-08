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
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'AI_UAV_Tests'))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GPS_Dropout_Recovery'))

import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

print("\n" + "="*80)
print("ADAPTIVE SEARCH MISSION - PRE-COMPENSATION HYBRID MODEL")
print("="*80)
print("\nUsing KALMAN FILTER + LSTM WAYPOINT PRE-COMPENSATION")
print("Real-time PyBullet visualization with randomized search patterns\n")

# =========================================================
# Pre-compensation constants
# =========================================================
SHADOW_RESET_INTERVAL_PRECOMP = 2500   # must match data collector
HORIZON_STRIDE_PRECOMP        = 100    # 100 steps x 0.005s = 0.5s per horizon
NUM_HORIZONS_PRECOMP           = 7     # +0.5s .. +3.5s
MAX_PRECOMP_HORIZON_S          = 3.5   # max LSTM coverage in seconds

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
from kalman_filter_ins import KalmanFilterINS

def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))

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
print("[3] Generating RANDOMIZED search mission...\n")

class AdaptiveSearchPatternGenerator:
    """Generate various search patterns with randomization."""
    
    def __init__(self, area_size=(10.0, 5.0), altitude=1.0):
        self.x_min, self.x_max = -area_size[0]/2, area_size[0]/2
        self.y_min, self.y_max = -area_size[1]/2, area_size[1]/2
        self.z = altitude
        self.area_size = area_size
        
    def zigzag_search(self, num_passes=4, randomness=0.15):
        waypoints = []
        x_width = self.x_max - self.x_min
        y_spacing = (self.y_max - self.y_min) / (num_passes + 1)
        
        for i in range(num_passes):
            y = self.y_min + (i + 1) * y_spacing
            x_start = self.x_min + np.random.uniform(-randomness, randomness) * x_width * 0.05
            x_end = self.x_max + np.random.uniform(-randomness, randomness) * x_width * 0.05
            x_start = np.clip(x_start, self.x_min, self.x_max)
            x_end = np.clip(x_end, self.x_min, self.x_max)
            
            num_intermediate = 2
            for j in range(num_intermediate + 1):
                x = x_start + (x_end - x_start) * j / num_intermediate
                y_varied = y + np.random.uniform(-randomness, randomness) * y_spacing * 0.3
                y_varied = np.clip(y_varied, self.y_min, self.y_max)
                waypoints.append(np.array([x, y_varied, self.z]))
            
            if i < num_passes - 1:
                y_turn = y + y_spacing * 0.5
                y_turn = np.clip(y_turn, self.y_min, self.y_max)
                waypoints.append(np.array([x_end, y_turn, self.z]))
        
        return np.array(waypoints)
    
    def spiral_search(self, rotations=3, randomness=0.2):
        waypoints = []
        num_points = 40
        
        for i in range(num_points):
            t = 2 * np.pi * rotations * i / num_points
            radius = (self.area_size[0] / 2) * i / num_points
            x = radius * np.cos(t) + np.random.uniform(-randomness, randomness) * self.area_size[0] * 0.1
            y = (radius / 1.5) * np.sin(t) + np.random.uniform(-randomness, randomness) * self.area_size[1] * 0.1
            x = np.clip(x, self.x_min, self.x_max)
            y = np.clip(y, self.y_min, self.y_max)
            waypoints.append(np.array([x, y, self.z]))
        
        return np.array(waypoints)
    
    def perimeter_search(self, num_laps=2, randomness=0.15):
        waypoints = []
        perimeter_length = 2 * (self.x_max - self.x_min) + 2 * (self.y_max - self.y_min)
        # ~2.4 waypoints/metre matches the training-data perimeter density (80wps/34m).
        pts_per_lap = max(8, int(round(perimeter_length * 2.4)))
        num_points = pts_per_lap * num_laps
        
        for i in range(num_points):
            fraction = (i % pts_per_lap) / pts_per_lap
            perimeter_pos = fraction * perimeter_length
            
            if perimeter_pos < self.x_max - self.x_min:
                x = self.x_min + perimeter_pos
                y = self.y_min
            elif perimeter_pos < 2 * (self.x_max - self.x_min):
                x = self.x_max
                y = self.y_min + (perimeter_pos - (self.x_max - self.x_min))
            elif perimeter_pos < 2 * (self.x_max - self.x_min) + (self.y_max - self.y_min):
                x = self.x_max - (perimeter_pos - 2 * (self.x_max - self.x_min))
                y = self.y_max
            else:
                x = self.x_min
                y = self.y_max - (perimeter_pos - 2 * (self.x_max - self.x_min) - (self.y_max - self.y_min))
            
            x += np.random.uniform(-randomness, randomness) * (self.x_max - self.x_min) * 0.1
            y += np.random.uniform(-randomness, randomness) * (self.y_max - self.y_min) * 0.1
            waypoints.append(np.array([x, y, self.z]))
        
        return np.array(waypoints)

generator = AdaptiveSearchPatternGenerator(area_size=(3.5, 2.5), altitude=1.0)
pattern_type = np.random.choice(['zigzag', 'spiral', 'perimeter'])
print(f"Generated pattern: {pattern_type.upper()}")

if pattern_type == 'zigzag':
    waypoints = generator.zigzag_search(num_passes=8, randomness=0.3)
elif pattern_type == 'spiral':
    waypoints = generator.spiral_search(rotations=4, randomness=0.2)
else:
    waypoints = generator.perimeter_search(num_laps=3, randomness=0.15)

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

TARGET_SPEED = 1.2
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
num_dropouts = int(T_final / 10)
dropout_windows = []
dropout_duration = 3.5

for i in range(num_dropouts):
    start_time = 3.0 + i * (T_final - 4.0) / max(num_dropouts, 1)
    end_time = start_time + dropout_duration
    if end_time < T_final - 1.0:
        dropout_windows.append((start_time, end_time))

print("GPS Dropout Windows:")
for i, (start, end) in enumerate(dropout_windows):
    print(f"  Window {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
print()

print("TIME   | GPS  | SOURCE           | EKF ERR | PRECOMP ERR | ALTITUDE | WPT")
print("-" * 80)

# State tracking
kf = KalmanFilterINS(dt=env.TIME_STEP)
kf.set_state(obs[0:3], env.drone.xyz_dot, env.drone.rpy)   # seed EKF
ekf_pos = obs[0:3].copy()
ekf_errors = []
precomp_errors = []   # error when using pre-compensated waypoints
prev_gps_ok = True
current_waypoint_idx = 0

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
            pos_for_control = true_pos
            vel_for_control = v_true
        else:
            # XY: EKF dead-reckoning; Z: barometric altitude (GPS-independent).
            pos_for_control = np.array([ekf_pos[0], ekf_pos[1], true_pos[2]])
            vel_for_control = v_true

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

        ctrl = quad.step(target_wp[:3] if path_active else env.get_mission_reference(),
                         np.zeros(3), z_ref=z_ref)
        rates_des = ctrl["rates_des"]
        U1 = ctrl["thrust_cmd"]

        # Build PID action
        pid_action = np.zeros(4, dtype=np.float32)
        pid_action[0] = thrust_to_action(U1, quad.m, quad.g)
        pid_action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

        # ---- Step physics engine ----
        obs, reward, done, truncated, info = env.step(pid_action)

        # ---- Read IMU gyro (always available; attitude is GPS-independent) ----
        imu_g = obs[10:13] if len(obs) > 12 else np.zeros(3)

        # ---- EKF: velocity dead-reckoning (NOT acceleration double-integration) ----
        kf.x[3:6] = v_true
        kf.x[6:9] = ang
        kf.predict()
        if gps_ok:
            kf.update_with_gps(obs[0:3])

        ekf_pos = kf.get_position()

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
        elif not gps_ok:
            steps_since_dropout += 1

        t_norm = min(steps_since_dropout / SHADOW_RESET_INTERVAL_PRECOMP, 1.0)

        # ---- Fill unified IMU buffer: [vel, gyro, t_norm] ----
        slot = buffer_idx % 300
        imu_buffer[slot] = np.concatenate([v_true, imu_g, [t_norm]])
        buffer_idx += 1

        # ---- LSTM pre-compensation inference (only during GPS dropout) ----
        if buffer_idx >= 300 and not gps_ok:
            # Roll circular buffer so oldest entry is first (matches training order)
            roll_by    = -(buffer_idx % 300)
            imu_ordered = np.roll(imu_buffer, roll_by, axis=0)
            imu_tensor  = torch.from_numpy(imu_ordered.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(imu_tensor)  # (1, 7, 3)

            # For each upcoming waypoint, estimate arrival time and select the
            # nearest LSTM horizon to predict drift at that moment.
            TARGET_SPEED = 1.2
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
                adjusted_waypoints[wi] = waypoints[wi] + predicted_drift
            # Refresh the visible green window (next N_SHOW only) and yellow indicator
            refresh_green_window(bc_client, adj_spheres, adjusted_waypoints,
                                 current_waypoint_idx, N_SHOW, _BURIED)
            if current_waypoint_idx < len(waypoints):
                move_sphere(bc_client, curr_sphere, adjusted_waypoints[current_waypoint_idx])

        # ---- Pre-compensation position error (for display only) ----
        # Measures EKF error relative to the pre-compensated target frame
        if current_waypoint_idx < len(waypoints):
            offset = adjusted_waypoints[current_waypoint_idx] - waypoints[current_waypoint_idx]
            error_precomp = np.linalg.norm((ekf_pos - offset) - true_pos)
        else:
            error_precomp = error_ekf
        precomp_errors.append(error_precomp)

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
            print(f"{t:6.1f}s| {gps_str} | {source_str} | {error_ekf*100:6.2f}cm | {error_precomp*100:8.2f}cm | {true_pos[2]:7.3f}m | {wpt_str}")

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

    print("Position Error Statistics:")
    print(f"  Pure EKF:              {ekf_mean*100:.2f} \u00b1 {ekf_std*100:.2f} cm")
    print(f"  EKF + Pre-Compensation:{pc_mean*100:.2f} \u00b1 {pc_std*100:.2f} cm")

    improvement = (pc_mean - ekf_mean) / ekf_mean * 100
    print(f"  Pre-Comp vs EKF:       {improvement:+.1f}%\n")

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

env.close()
