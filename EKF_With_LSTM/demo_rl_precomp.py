#!/usr/bin/env python3
"""
RL Pre-Compensation Mission Demo
=================================
Demonstrates the trained PPO residual correction policy on top of the frozen
LSTM waypoint pre-compensation system.

Shows three comparison columns:
  EKF ERR    — pure dead-reckoning error (no compensation)
  PRECOMP ERR — EKF + LSTM pre-compensation only (no RL)
  RL ERR     — EKF + LSTM + RL residual correction (this demo)

Load order:
  1. LSTM  : experiments/lstm_position_drift_precomp.pt   (frozen)
  2. Policy: experiments/rl_precomp_policy.zip            (trained PPO)
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
from pathlib import Path

print("\n" + "=" * 80)
print("ADAPTIVE SEARCH MISSION — RL PRE-COMPENSATION DEMO")
print("=" * 80)
print("\nEKF + Frozen LSTM + PPO Residual Correction Policy\n")

# ── Constants (must match uav_rl_env.py / precomp pipeline) ──────────────────
SHADOW_RESET_INTERVAL = 2500
HORIZON_STRIDE        = 100
NUM_HORIZONS          = 7
WAYPOINT_TOLERANCE    = 0.5
TARGET_SPEED          = 1.2
TAKEOFF_Z             = 0.16
PATHFOLLOW_TRIGGER_Z  = 0.15
BUFFER_LEN            = 300
PLAYBACK_SLOWDOWN     = 1.0
MAX_WAYPOINT_SHIFT    = 0.25    # metres — hard cap on adjusted waypoint displacement (V11)

LSTM_MODEL_PATH   = Path('experiments/lstm_position_drift_precomp.pt')
POLICY_MODEL_PATH = Path('experiments/rl_precomp_policy_v8.zip')

# ── LSTM (same class as training) ─────────────────────────────────────────────
class PositionDriftLSTMPrecomp(nn.Module):
    def __init__(self, horizon_steps=7, hidden_size=256):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.proj = nn.Linear(7, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3)
            )
            for _ in range(horizon_steps)
        ])

    def forward(self, x):
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        return torch.stack([h(final) for h in self.heads], dim=1)  # (B, 7, 3)


# ── Load models ───────────────────────────────────────────────────────────────
print("[1] Loading models...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not LSTM_MODEL_PATH.exists():
    print(f"[ERROR] LSTM model not found: {LSTM_MODEL_PATH}")
    sys.exit(1)
lstm_model = PositionDriftLSTMPrecomp(NUM_HORIZONS, 256).to(device)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True))
lstm_model.eval()
print(f"    [OK] LSTM loaded  ({LSTM_MODEL_PATH.name})")

if not POLICY_MODEL_PATH.exists():
    print(f"[ERROR] RL policy not found: {POLICY_MODEL_PATH}")
    print("  Run: python train_rl_precomp.py")
    sys.exit(1)

from stable_baselines3 import PPO
rl_policy = PPO.load(str(POLICY_MODEL_PATH), device='cpu')
print(f"    [OK] RL policy loaded  ({POLICY_MODEL_PATH.name})\n")

# ── Environment ───────────────────────────────────────────────────────────────
print("[2] Creating PyBullet environment...")
from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
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

# ── Helper: draw tracer line ──────────────────────────────────────────────────
def draw_line(bc, start, end, color, width=2):
    bc.addUserDebugLine(lineFromXYZ=start, lineToXYZ=end,
                        lineColorRGB=color, lineWidth=width)

# ── Waypoint / mission generation (same as precomp demo) ─────────────────────
print("[3] Generating randomised search mission...\n")

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

print("[4] Starting RL pre-compensation mission...\n")
print(f"Mission Duration: {T_final:.1f}s  |  Distance: {total_dist:.1f}m at {TARGET_SPEED}m/s")
print("GPS Dropout Windows:")
for i, (s, e) in enumerate(dropout_windows):
    print(f"  Window {i+1}: {s:.1f}s – {e:.1f}s  ({e-s:.1f}s)")
print()

print("TIME   | GPS  | SOURCE      | EKF ERR | PRECOMP ERR | RL ERR   | AX    | AY    | ALT    | WPT")
print("-" * 101)

# ── State ─────────────────────────────────────────────────────────────────────
kf      = KalmanFilterINS(dt=env.TIME_STEP)
kf.set_state(obs[0:3], env.drone.xyz_dot, env.drone.rpy)
ekf_pos = obs[0:3].copy()

ekf_errors    = []
precomp_errors = []
rl_errors     = []

prev_gps_ok         = True
current_waypoint_idx = 0
reached_waypoints   = 0
steps_since_dropout = 0
adjusted_waypoints  = waypoints.copy()    # RL+LSTM adjusted (what drone follows)
lstm_only_waypoints = waypoints.copy()     # LSTM-only adjusted (counterfactual for display)
imu_buffer          = np.zeros((BUFFER_LEN, 7), dtype=np.float32)
buffer_idx          = 0
last_lstm_preds     = np.zeros((NUM_HORIZONS, 3), dtype=np.float32)

# Tracers
position_history    = []
last_tracer_step    = 0
TRACER_INTERVAL     = 10


def build_obs(lstm_preds, t_norm, v_true, ekf_pos, waypoints, wp_idx, gps_ok_flag,
              kf_P, adjusted_waypoints, lstm_only_waypoints):
    """Build the 42-dim observation vector (must match UAVPrecompEnv._get_obs)."""
    lstm_flat  = np.clip(lstm_preds.flatten(), -1.0, 1.0).astype(np.float32)
    vel        = np.clip(v_true, -5.0, 5.0).astype(np.float32)
    if wp_idx < len(waypoints):
        diff       = waypoints[wp_idx][:2] - ekf_pos[:2]
        dist       = float(np.linalg.norm(diff))
        wp_dir     = (diff / dist if dist > 1e-3 else np.zeros(2)).astype(np.float32)
        dist_to_wp = np.float32(min(dist, 5.0))
    else:
        wp_dir, dist_to_wp = np.zeros(2, dtype=np.float32), np.float32(0.0)
    gps_flag = np.float32(1.0 if gps_ok_flag else 0.0)
    # [29:32] EKF XYZ position std-dev
    ekf_pos_std = np.sqrt(np.clip(np.diag(kf_P)[:3], 0.0, None))
    ekf_pos_std = np.clip(ekf_pos_std, 0.0, 5.0).astype(np.float32)
    # [32:34] applied alpha per axis (centred: alpha-1)
    if (adjusted_waypoints is not None and lstm_only_waypoints is not None
            and wp_idx < len(waypoints)):
        applied_alpha_xy = np.clip(
            adjusted_waypoints[wp_idx][:2] - lstm_only_waypoints[wp_idx][:2],
            -1.0, 1.0
        ).astype(np.float32)
    else:
        applied_alpha_xy = np.zeros(2, dtype=np.float32)
    # [34:40] Lookahead segment direction vectors; [40:42] segment lengths
    n_wps = len(waypoints)
    lookahead_vecs = np.zeros(6, dtype=np.float32)
    lookahead_lens = np.zeros(2, dtype=np.float32)
    for j in range(3):
        a, b = wp_idx + j, wp_idx + j + 1
        if b < n_wps:
            seg = waypoints[b][:2] - waypoints[a][:2]
            seg_len = float(np.linalg.norm(seg))
            lookahead_vecs[j*2:j*2+2] = (
                (seg / seg_len).astype(np.float32) if seg_len > 1e-3
                else np.zeros(2, dtype=np.float32)
            )
            if j < 2:
                lookahead_lens[j] = np.float32(min(seg_len, 5.0))
    return np.concatenate([lstm_flat, [np.float32(t_norm)], vel, wp_dir,
                           [dist_to_wp], [gps_flag], ekf_pos_std, applied_alpha_xy,
                           lookahead_vecs, lookahead_lens])


# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    for k in range(steps):
        time.sleep(dt * PLAYBACK_SLOWDOWN)

        env.mission_time += dt
        t = env.mission_time

        gps_ok = not any(s <= t < e for s, e in dropout_windows)

        true_pos = env.drone.xyz.copy()
        v_true   = env.drone.xyz_dot.copy()
        ang      = env.drone.rpy.copy()
        rate     = env.drone.rpy_dot.copy()

        if not path_active and true_pos[2] > PATHFOLLOW_TRIGGER_Z:
            print(f"[*] Path following activated at t={t:.1f}s, z={true_pos[2]:.3f}m")
            path_active = True

        if gps_ok:
            pos_for_control = true_pos.copy()
            vel_for_control = v_true.copy()
        else:
            pos_for_control = np.array([ekf_pos[0], ekf_pos[1], true_pos[2]])
            vel_for_control = v_true.copy()

        # Waypoint following (uses RL+LSTM adjusted targets)
        if path_active and current_waypoint_idx < len(waypoints):
            target_wp  = adjusted_waypoints[current_waypoint_idx]
            if np.linalg.norm(pos_for_control - target_wp) < WAYPOINT_TOLERANCE:
                reached_waypoints    += 1
                current_waypoint_idx += 1
                if current_waypoint_idx < len(waypoints):
                    target_wp = adjusted_waypoints[current_waypoint_idx]
        else:
            target_wp = adjusted_waypoints[-1] if len(waypoints) > 0 else np.array([0, 0, 1.0])

        # PID
        quad.inject_external_state(pos_for_control, vel_for_control, ang, rate)
        z_ref  = env.get_mission_reference()[2] if not path_active else target_wp[2]
        ctrl   = quad.step(target_wp[:3] if path_active else env.get_mission_reference(),
                           np.zeros(3), z_ref=z_ref)
        pid_action    = np.zeros(4, dtype=np.float32)
        pid_action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        pid_action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(pid_action)

        imu_g = obs[10:13] if len(obs) > 12 else np.zeros(3)

        # EKF
        kf.x[3:6] = v_true
        kf.x[6:9] = ang
        kf.predict()
        if gps_ok:
            kf.update_with_gps(obs[0:3])
        ekf_pos = kf.get_position().copy()

        error_ekf = float(np.linalg.norm(ekf_pos - true_pos))
        ekf_errors.append(error_ekf)

        # t_norm
        if not prev_gps_ok and gps_ok:
            steps_since_dropout = 0
            adjusted_waypoints  = waypoints.copy()
            lstm_only_waypoints = waypoints.copy()
        elif not gps_ok:
            steps_since_dropout += 1
        t_norm = min(steps_since_dropout / SHADOW_RESET_INTERVAL, 1.0)

        # IMU buffer
        slot = buffer_idx % BUFFER_LEN
        imu_buffer[slot] = np.concatenate([v_true, imu_g, [t_norm]])
        buffer_idx += 1

        # LSTM inference + RL action (dropout only)
        rl_action_mag = 0.0
        alpha_x = 1.0  # default: pure LSTM scaling (per-axis)
        alpha_y = 1.0
        if buffer_idx >= BUFFER_LEN and not gps_ok:
            roll_by     = -(buffer_idx % BUFFER_LEN)
            imu_ordered = np.roll(imu_buffer, roll_by, axis=0)
            tensor      = torch.from_numpy(imu_ordered.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = lstm_model(tensor)
            last_lstm_preds = preds[0].cpu().numpy()

            # Build observation for RL policy
            rl_obs = build_obs(last_lstm_preds, t_norm, v_true, ekf_pos,
                               waypoints, current_waypoint_idx, gps_ok,
                               kf.P, adjusted_waypoints, lstm_only_waypoints)
            rl_action, _ = rl_policy.predict(rl_obs, deterministic=True)
            alpha_x = float(rl_action[0])
            alpha_y = float(rl_action[1])
            rl_action_mag = abs(alpha_x - 1.0) + abs(alpha_y - 1.0)  # total deviation

            for wi in range(current_waypoint_idx,
                            min(current_waypoint_idx + 5, len(waypoints))):
                dist    = float(np.linalg.norm(ekf_pos - waypoints[wi]))
                t_arr_s = dist / TARGET_SPEED if dist > 1e-3 else 0.0
                h_idx   = min(int(t_arr_s / (HORIZON_STRIDE * dt)), NUM_HORIZONS - 1)

                drift = last_lstm_preds[h_idx].copy()
                drift[2] = 0.0  # XY only

                # Project correction onto approach direction — prevents cross-track
                # component from pushing turn waypoints past boundaries (zigzag fix)
                approach_xy = waypoints[wi][:2] - ekf_pos[:2]
                a_norm      = np.linalg.norm(approach_xy)
                if a_norm > 1e-3:
                    u           = approach_xy / a_norm
                    drift_along = np.dot(drift[:2], u) * u          # parallel only
                else:
                    drift_along = drift[:2]

                lstm_only_waypoints[wi] = waypoints[wi] + np.array([drift_along[0], drift_along[1], 0.0])
                shift_xy  = np.array([alpha_x * drift_along[0], alpha_y * drift_along[1]])
                shift_mag = np.linalg.norm(shift_xy)
                if shift_mag > MAX_WAYPOINT_SHIFT:        # V11: cap displacement
                    shift_xy = shift_xy * (MAX_WAYPOINT_SHIFT / shift_mag)
                adjusted_waypoints[wi]  = waypoints[wi] + np.array([shift_xy[0], shift_xy[1], 0.0])
        else:
            last_lstm_preds = np.zeros((NUM_HORIZONS, 3), dtype=np.float32)

        # ── Error metrics ────────────────────────────────────────────────────
        # PRECOMP ERR: how far ekf_pos (corrected for LSTM offset) is from truth
        if current_waypoint_idx < len(waypoints):
            lstm_offset   = lstm_only_waypoints[current_waypoint_idx] - waypoints[current_waypoint_idx]
            error_precomp = float(np.linalg.norm((ekf_pos - lstm_offset) - true_pos))

            rl_offset  = adjusted_waypoints[current_waypoint_idx] - waypoints[current_waypoint_idx]
            error_rl   = float(np.linalg.norm((ekf_pos - rl_offset) - true_pos))
        else:
            error_precomp = error_ekf
            error_rl      = error_ekf

        precomp_errors.append(error_precomp)
        rl_errors.append(error_rl)

        position_history.append((true_pos.copy(), gps_ok))

        # Tracers
        if k > TRACER_INTERVAL and (k - last_tracer_step) >= TRACER_INTERVAL:
            prev_idx = len(position_history) - TRACER_INTERVAL - 1
            curr_idx = len(position_history) - 1
            if 0 <= prev_idx < len(position_history):
                prev_pos, prev_gps = position_history[prev_idx]
                curr_pos, curr_gps = position_history[curr_idx]
                color = (0, 0.6, 1) if curr_gps else (1, 0.2, 0)
                draw_line(bc_client, prev_pos, curr_pos, color, width=2)
            last_tracer_step = k

        prev_gps_ok = gps_ok

        if k % 50 == 0:
            gps_str    = "ON " if gps_ok else "OFF"
            src_str    = "GPS    " if gps_ok else ("RL+PRECOMP" if buffer_idx >= BUFFER_LEN else "EKF    ")
            wpt_str    = f"{current_waypoint_idx}/{len(waypoints)}"
            ax_disp = (alpha_x if buffer_idx >= BUFFER_LEN and not gps_ok else 1.0)
            ay_disp = (alpha_y if buffer_idx >= BUFFER_LEN and not gps_ok else 1.0)
            print(f"{t:6.1f}s| {gps_str} | {src_str:10s} | "
                  f"{error_ekf*100:6.2f}cm | {error_precomp*100:8.2f}cm | "
                  f"{error_rl*100:6.2f}cm | {ax_disp:5.3f} | {ay_disp:5.3f} | "
                  f"{true_pos[2]:5.3f}m | {wpt_str}")

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
    ekf_mean  = np.mean(ekf_errors)
    ekf_std   = np.std(ekf_errors)
    pc_mean   = np.mean(precomp_errors)
    pc_std    = np.std(precomp_errors)
    rl_mean   = np.mean(rl_errors)
    rl_std    = np.std(rl_errors)

    print("Position Error Statistics:")
    print(f"  Pure EKF              : {ekf_mean*100:.2f} ± {ekf_std*100:.2f} cm")
    print(f"  EKF + LSTM PreComp    : {pc_mean*100:.2f} ± {pc_std*100:.2f} cm  "
          f"({(pc_mean-ekf_mean)/ekf_mean*100:+.1f}% vs EKF)")
    print(f"  EKF + LSTM + RL       : {rl_mean*100:.2f} ± {rl_std*100:.2f} cm  "
          f"({(rl_mean-ekf_mean)/ekf_mean*100:+.1f}% vs EKF, "
          f"{(rl_mean-pc_mean)/pc_mean*100:+.1f}% vs LSTM-only)")
    print()

print("Models:")
print(f"  LSTM  : {LSTM_MODEL_PATH.name}")
print(f"  Policy: {POLICY_MODEL_PATH.name}\n")

print("=" * 80)
print("[SUCCESS] RL pre-compensation demo complete!")
print("=" * 80)
