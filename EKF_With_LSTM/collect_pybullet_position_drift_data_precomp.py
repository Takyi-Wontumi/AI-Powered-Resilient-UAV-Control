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
  - Uses the EXACT same DroneMissionEnv + QuadcopterPID + KalmanFilterINS stack as
    demo_adaptive_search_v3_hybrid_mission_gui.py
  - Records env.drone.xyz_dot and obs[10:13] — the same signals stored in the
    demo's imu_buffer_a / imu_buffer_g at inference time
  - Runs the same 3.5×2.5m perimeter/zigzag/spiral patterns as the demo
  - Measures dead-reckoning drift via a "shadow" KF that never receives GPS

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

# Force PyBullet into DIRECT (no-GUI) mode before any import
os.environ['PYBULLET_USE_SHARED_MEMORY'] = '0'

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'AI_UAV_Tests'))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GPS_Dropout_Recovery'))

import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

print("\n" + "="*100)
print("COLLECT POSITION-DRIFT TRAINING DATA FROM PYBULLET")
print("="*100 + "\n")

# =========================================================
# Configuration
# =========================================================
NUM_MISSIONS     = 600    # target missions — high-volume run
STEPS_PER_MISSION = 6000  # 30s at 200 Hz — needed for 2500-step reset windows
SHADOW_RESET_INTERVAL = 2500  # re-anchor shadow KF every 12.5s — covers 10s+ dropouts
SAVE_DIR         = Path('data/position_drift')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Pattern distribution: heavily oversample zigzag + perimeter (corner-heavy).
# Hover added so model learns zero-velocity = zero-drift.
# Spiral is already well-learnt; minimal allocation.
PATTERNS         = ['perimeter', 'zigzag', 'spiral', 'hover']
PATTERN_WEIGHTS  = [0.40, 0.40, 0.10, 0.10]

# =========================================================
# Imports
# =========================================================
print("[1/4] Loading dependencies...")
try:
    from phoenix_drone_simulation.envs.control import AttitudeRate
    from phoenix_drone_simulation.envs.mission import DroneMissionEnv
    from quadcopter_env import QuadcopterPID
    from kalman_filter_ins import KalmanFilterINS
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Ensure phoenix_drone_simulation, quadcopter_env, kalman_filter_ins are on sys.path")
    sys.exit(1)
print("    [OK]\n")


# =========================================================
# Search-pattern generator  (matches demo exactly)
# =========================================================
class AdaptiveSearchPatternGenerator:
    """Generates same waypoint patterns as the demo (3.5×2.5m field)."""

    def __init__(self, area_size=(3.5, 2.5), altitude=1.0):
        self.x_min = -area_size[0] / 2
        self.x_max =  area_size[0] / 2
        self.y_min = -area_size[1] / 2
        self.y_max =  area_size[1] / 2
        self.z     = altitude
        self.area_size = area_size

    def perimeter_search(self, num_laps=3, randomness=0.12):
        """Dense rectangular perimeter — same as demo perimeter_search.
        50% of calls run clockwise (wz<0 at corners) for chirality diversity."""
        wps = []
        perim = 2*(self.x_max-self.x_min) + 2*(self.y_max-self.y_min)
        ppl   = max(8, int(round(perim * 2.4)))   # ~29 pts/lap on 3.5x2.5m
        dx, dy = self.x_max-self.x_min, self.y_max-self.y_min
        for i in range(ppl * num_laps):
            frac = (i % ppl) / ppl
            p    = frac * perim
            if   p < dx:          x, y = self.x_min + p,        self.y_min
            elif p < dx + dy:     x, y = self.x_max,             self.y_min + (p - dx)
            elif p < 2*dx + dy:   x, y = self.x_max - (p-dx-dy), self.y_max
            else:                 x, y = self.x_min,             self.y_max - (p-2*dx-dy)
            x += np.random.uniform(-randomness, randomness) * dx * 0.1
            y += np.random.uniform(-randomness, randomness) * dy * 0.1
            wps.append(np.array([np.clip(x, self.x_min, self.x_max),
                                  np.clip(y, self.y_min, self.y_max),
                                  self.z]))
        if np.random.random() < 0.5:   # flip to clockwise 50% of the time
            wps = wps[::-1]
        return np.array(wps)

    def zigzag_search(self, num_passes=8, randomness=0.15):
        """Lawnmower zigzag — same as demo zigzag_search.
        50% of calls sweep y in reverse (south-to-north vs north-to-south)
        and 50% flip the initial row direction for full turn-signature diversity."""
        wps = []
        y_spacing = (self.y_max - self.y_min) / (num_passes + 1)
        reverse_y    = np.random.random() < 0.5  # sweep direction: low->high or high->low
        flip_x_start = np.random.random() < 0.5  # which end starts first
        for i in range(num_passes):
            yi = (num_passes - 1 - i) if reverse_y else i
            y  = self.y_min + (yi + 1) * y_spacing
            even = (i % 2 == 0) if not flip_x_start else (i % 2 != 0)
            xs = self.x_min if even else self.x_max
            xe = self.x_max if even else self.x_min
            for j in range(3):   # 3 intermediate points per row
                x   = xs + (xe - xs) * j / 2
                yv  = np.clip(y + np.random.uniform(-randomness, randomness) * y_spacing * 0.3,
                              self.y_min, self.y_max)
                wps.append(np.array([x, yv, self.z]))
            if i < num_passes - 1:
                wps.append(np.array([xe, np.clip(y + y_spacing*0.5, self.y_min, self.y_max), self.z]))
        return np.array(wps)

    def spiral_search(self, rotations=4, randomness=0.1):
        """Expanding spiral — same as demo spiral_search.
        50% of calls run clockwise (negative angular velocity) for chirality diversity."""
        wps = []
        chirality = 1.0 if np.random.random() < 0.5 else -1.0  # +1=CCW, -1=CW
        for i in range(40):
            t      = chirality * 2 * np.pi * rotations * i / 40
            radius = (self.area_size[0] / 2) * i / 40
            x = np.clip(radius*np.cos(t) + np.random.randn()*randomness*0.3,
                        self.x_min, self.x_max)
            y = np.clip((radius/1.5)*np.sin(t) + np.random.randn()*randomness*0.3,
                        self.y_min, self.y_max)
            wps.append(np.array([x, y, self.z]))
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
        if pattern == 'perimeter':
            wps = generator.perimeter_search(
                num_laps=np.random.randint(2, 4),
                randomness=np.random.uniform(0.05, 0.20))
        elif pattern == 'zigzag':
            wps = generator.zigzag_search(
                num_passes=np.random.randint(5, 10),
                randomness=np.random.uniform(0.05, 0.25))
        elif pattern == 'spiral':
            wps = generator.spiral_search(
                rotations=np.random.randint(3, 6),
                randomness=np.random.uniform(0.05, 0.15))
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

    kf_ctrl   = KalmanFilterINS(dt=env.TIME_STEP)
    kf_shadow = KalmanFilterINS(dt=env.TIME_STEP)

    pos0 = obs[0:3].copy()
    vel0 = env.drone.xyz_dot.copy()
    att0 = env.drone.rpy.copy()

    kf_ctrl.set_state(pos0, vel0, att0)
    kf_shadow.set_state(pos0, vel0, att0)

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

        # ---- Gyro from observation (same index as the demo) ----
        imu_g = obs[10:13] if len(obs) > 12 else rate

        # ---- Control KF: always has GPS — clean position for bookkeeping ----
        kf_ctrl.x[3:6] = v_true
        kf_ctrl.x[6:9] = ang
        kf_ctrl.predict()
        kf_ctrl.update_with_gps(obs[0:3])

        # ---- Shadow KF: NEVER gets GPS — accumulates dead-reckoning drift ----
        # Simulates exactly what the demo's EKF does during a GPS dropout window.
        # Uses the same velocity injection as the demo (v_true → kf.x[3:6]).
        kf_shadow.x[3:6] = v_true
        kf_shadow.x[6:9] = ang
        kf_shadow.predict()

        # Re-anchor shadow KF to true position every SHADOW_RESET_INTERVAL steps.
        # This simulates a "GPS just recovered and is about to drop out again",
        # giving us many short windows of realistic drift growth within one mission.
        if step > 0 and step % SHADOW_RESET_INTERVAL == 0:
            kf_shadow.set_state(obs[0:3], v_true, ang)

        # Compute time-since-reset feature (normalised 0->1 over one reset window).
        # This tells the model how far into a dropout window it currently is.
        steps_since_reset = step % SHADOW_RESET_INTERVAL
        t_norm = steps_since_reset / SHADOW_RESET_INTERVAL

        # ---- Record (these are exactly the demo's imu_buffer_a and imu_buffer_g) ----
        vel_list.append(v_true.copy())
        gyro_list.append(imu_g.copy())
        true_list.append(obs[0:3].copy())
        dr_list.append(kf_shadow.get_position().copy())
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
        'dr_pos':         np.array(dr_list,    dtype=np.float32),   # (T, 3)
        't_norm_meas':    np.array(t_norm_list, dtype=np.float32),  # (T,)  time-since-reset 0->1
        'mission_type':   pattern,
        'duration_steps': n,
        'source':         'pybullet_precomp',
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
print(f"\nNext: run train_lstm_position_drift.py to retrain on combined dataset.")
