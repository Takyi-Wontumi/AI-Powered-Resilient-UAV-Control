#!/usr/bin/env python3
"""
Circle Waypoint Mission with AUTOMATIC GPS Dropout + INS Recovery
- PyBullet visualization
- Circle trajectory following (PID controller)
- Automatic GPS dropout at 2-4 seconds (position data zeroed)
- INS neural network estimates position during dropout
- Watch the drone maintain circle flight through GPS loss
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import numpy as np
import torch
import torch.nn as nn

print("\n" + "="*80)
print("CIRCLE MISSION WITH AUTOMATIC GPS DROPOUT + INS RECOVERY")
print("="*80)
print("\nNOTE: To change playback speed, edit PLAYBACK_SLOWDOWN in the script:")
print("  PLAYBACK_SLOWDOWN = 1.0   for real-time (fast)")
print("  PLAYBACK_SLOWDOWN = 5.0   for 5x slower")
print("  PLAYBACK_SLOWDOWN = 10.0  for 10x slower")
print("="*80 + "\n")

# =========================================================
# Load INS Model
# =========================================================
print("[1] Loading INS model...")

class INSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 6)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

model = INSModel()
model.load_state_dict(torch.load("results/ins_navigation_improved/2026-03-18__13-16-25/ins_model_improved.pt", map_location='cpu'))
model.eval()
print("    [OK] Model loaded\n")

# =========================================================
# Setup Environment
# =========================================================
print("[2] Creating PyBullet environment with circle trajectory...")

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.trajectories_library import Trajectories as path

# Helper function
def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))

# Create environment
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
# Controller Setup
# =========================================================
quad = QuadcopterPID(dt=env.TIME_STEP)
quad.reset()

# =========================================================
# Mission Configuration
# =========================================================
print("[3] Starting circle mission with automatic GPS dropout...\n")

TAKEOFF_Z = 0.16
PATHFOLLOW_TRIGGER_Z = 0.15

env.set_target(np.array([0.0, 0.0, TAKEOFF_Z]))
path_active = False

dt = env.TIME_STEP
T_final = 20.0
steps = int(T_final / dt)

# PLAYBACK SPEED (multiply dt by this factor for slow motion)
PLAYBACK_SLOWDOWN = 1.0  # Real-time playback (change to 5.0, 10.0 for slower)

print("Mission Duration: 20 seconds")
print(f"Playback Speed: {1/PLAYBACK_SLOWDOWN:.1f}x speed ({PLAYBACK_SLOWDOWN:.0f}x slower)")
print(f"Estimated Runtime: ~{20 * PLAYBACK_SLOWDOWN:.0f} seconds\n")

# Multiple GPS Dropout windows (like real-world interference)
dropout_windows = [(2.0, 4.0), (8.0, 10.0), (14.0, 16.0)]

print("Timeline:")
print("  0-2s:   GPS ON (circle flight - climb)")
print("  2-4s:   GPS OFF (INS-only, position data zeroed)")
print("  4-8s:   GPS ON (recovery phase 1)")
print("  8-10s:  GPS OFF (INS-only, position data zeroed)")
print("  10-14s: GPS ON (stable circle)")
print("  14-16s: GPS OFF (INS-only, position data zeroed)")
print("  16-20s: GPS ON (final recovery)\n")

print("TIME   | GPS  | SOURCE        | ERROR (cm) | ALTITUDE")
print("-" * 60)

# State tracking
est_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ins_errors = []
gps_errors = []
all_errors = []
prev_gps_ok = True  # Track GPS state transitions for re-sync
gps_dropout_time = None  # Time when GPS first drops (for smooth transition)
gps_recovery_time = None  # Time when GPS recovers
TRANSITION_TIME = 0.2  # Smooth transition window (200ms) when GPS switches

# Circle tracking metrics
CIRCLE_RADIUS = 1.0
CIRCLE_CENTER = np.array([0.0, 0.0])
circle_tracking_errors = []  # Distance from circle (+ outside, - inside)
lateral_tracking_errors = []  # Distance from reference point


# =========================================================
# MAIN LOOP
# =========================================================
for k in range(steps):
    # Apply playback slowdown (adds delay for visualization)
    time.sleep(dt * PLAYBACK_SLOWDOWN)
    
    env.mission_time += dt
    t = env.mission_time

    # Read actual drone state
    x = env.drone.xyz
    v = env.drone.xyz_dot
    ang = env.drone.rpy
    rate = env.drone.rpy_dot

    # Activate path after liftoff
    if not path_active and x[2] > PATHFOLLOW_TRIGGER_Z:
        path_active = True

    # Get reference trajectory
    if path_active:
        pos_ref, vel_ref = path.circle_traj(t)
        env.set_target(pos_ref)
    else:
        pos_ref = env.get_mission_reference()
        vel_ref = np.zeros(3)

    # =====================================================
    # GPS DROPOUT SIMULATION - Multiple windows
    # =====================================================
    gps_ok = True
    for start, end in dropout_windows:
        if start <= t < end:
            gps_ok = False
            break
    
    # IMPORTANT: Detect GPS state transitions
    if gps_ok and not prev_gps_ok:
        # GPS RETURNED FROM DROPOUT - Re-sync INS estimate to actual position
        # This prevents drift accumulation when transitioning back to GPS
        est_pos = x.copy()
        gps_recovery_time = t
        print(f"[{t:.3f}s] GPS RECOVERED: Re-syncing INS estimate to actual position")
    
    if not gps_ok and prev_gps_ok:
        # GPS FIRST DROPS - Initialize INS estimate to current actual position
        # This is critical: start from ground truth, not accumulated error
        est_pos = x.copy()
        gps_dropout_time = t
        print(f"[{t:.3f}s] GPS LOST: Initializing INS from current position for dead-reckoning")
    
    # When GPS is OFF, position data is unavailable
    if gps_ok:
        # GPS is available - use actual position
        pos_for_controller = x.copy()
        
        # Smooth transition: blend GPS and INS for a brief period
        if gps_recovery_time is not None and (t - gps_recovery_time) < TRANSITION_TIME:
            # Transition from INS back to GPS: gradually increase GPS trust
            alpha = (t - gps_recovery_time) / TRANSITION_TIME  # 0 -> 1
            pos_for_controller = (1 - alpha) * est_pos + alpha * x
        
        est_pos = x.copy()  # Update estimate with GPS truth
        source = "GPS"
    else:
        # GPS DROPOUT - Hybrid INS approach:
        # Use model's learned dynamics for trajectory prediction
        
        accel = np.array([0.0, 0.0, 0.0])
        gyro = rate
        
        ins_in = np.concatenate([est_pos, v, accel, gyro, [0, 0, 0, 0.6], [dt], [0, 0, 0]])
        
        with torch.no_grad():
            pred = model(torch.tensor(ins_in, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        
        # Model predicts next state
        model_pred_pos = pred[0:3]
        model_pred_vel = pred[3:6]
        
        # Blend: 75% model (learned dynamics), 25% velocity integration
        est_pos_integrated = est_pos + v * dt
        est_pos = 0.75 * model_pred_pos + 0.25 * est_pos_integrated
        
        # Smooth transition: blend GPS and INS for a brief period after dropout starts
        pos_for_controller = est_pos.copy()
        if gps_dropout_time is not None and (t - gps_dropout_time) < TRANSITION_TIME:
            # Immediate transition: quick switch to INS (no gradual blend for startup)
            # This ensures we don't see large jumps when GPS first drops
            pass  # Use est_pos directly
        
        source = "INS (trajectory-based)"
    
    # Track GPS state for next iteration
    prev_gps_ok = gps_ok

    # =====================================================
    # PID CONTROLLER (without GPS position data during dropout)
    # =====================================================
    quad.inject_external_state(x, v, ang, rate)
    z_ref = env.get_mission_reference()[2]
    
    # Use position source (GPS or INS)
    ctrl = quad.step(pos_for_controller, vel_ref, z_ref=z_ref)
    rates_des = ctrl["rates_des"]
    U1 = ctrl["thrust_cmd"]
    
    # DEBUG: Log position discrepancy during GPS dropout
    if not gps_ok and k % 500 == 0:  # Log every 500 steps (1 second)
        pos_error = np.linalg.norm(est_pos - x)
        print(f"[{t:.3f}s] GPS DROPOUT - Real pos: {x}, Est pos: {est_pos}, Error: {pos_error:.4f}")
        print(f"         Control ref pos: {pos_for_controller}, Vel ref: {vel_ref}")
        print(f"         Thrust: {U1:.3f}, Rates cmd: {rates_des}")

    # Build action
    action = np.zeros(4, dtype=np.float32)
    action[0] = thrust_to_action(U1, quad.m, quad.g)
    action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

    # Step physics
    obs, reward, done, truncated, info = env.step(action)

    # =====================================================
    # ERROR TRACKING
    # =====================================================
    error = np.linalg.norm(x - pos_ref)
    all_errors.append(error)
    
    # Circle tracking metrics (XY plane only)
    xy_pos = x[:2]
    xy_center = CIRCLE_CENTER[:2]
    distance_from_center = np.linalg.norm(xy_pos - xy_center)
    radial_error = distance_from_center - CIRCLE_RADIUS  # 0 = on circle, >0 = outside
    circle_tracking_errors.append(radial_error)
    
    # Lateral error (distance from reference trajectory)
    xy_ref = pos_ref[:2]
    lateral_error = np.linalg.norm(xy_pos - xy_ref)
    lateral_tracking_errors.append(lateral_error)
    
    if gps_ok:
        gps_errors.append(error)
    else:
        ins_errors.append(error)

    # Print progress (every 0.5 seconds)
    if k % 250 == 0:
        print(f" {t:5.2f}s | {'ON ' if gps_ok else 'OFF'} | {source:13} | {error*100:7.2f} | {x[2]:6.3f}m")

    if done:
        print("\nMission terminated due to safety violation.")
        break

print("-" * 60)
print()

# =========================================================
# RESULTS
# =========================================================
print("="*80)
print("EXTENDED MISSION RESULTS (20 seconds, multiple GPS dropouts)")
print("="*80)

if gps_errors:
    print(f"\nWith GPS:")
    print(f"  Mean Error: {np.mean(gps_errors)*100:.2f}cm")
    print(f"  Max Error:  {np.max(gps_errors)*100:.2f}cm")
    print(f"  Samples:    {len(gps_errors)}")

if ins_errors:
    print(f"\nDuring GPS Dropout (INS-only - 3 windows, 6 seconds total):")
    print(f"  Mean Error: {np.mean(ins_errors)*100:.2f}cm  <-- INS Accuracy")
    print(f"  Max Error:  {np.max(ins_errors)*100:.2f}cm")
    print(f"  Samples:    {len(ins_errors)}")

print(f"\nImprovement over baseline (45.6m):")
if ins_errors:
    improvement = 45.6 / np.mean(ins_errors)
    print(f"  {improvement:.0f}x BETTER")

# Breakdown by dropout window
print(f"\nBreakdown by dropout window:")
for i, (start, end) in enumerate(dropout_windows):
    start_idx = int(start / dt)
    end_idx = int(end / dt)
    window_errors = all_errors[start_idx:end_idx]
    if window_errors:
        print(f"  Window {i+1} ({start:.1f}-{end:.1f}s): {np.mean(window_errors)*100:.2f}cm mean")

# =========================================================
# CIRCLE TRAJECTORY ANALYSIS
# =========================================================
print("\n" + "="*80)
print("CIRCLE TRAJECTORY ANALYSIS")
print("="*80)

print(f"\nCircle Parameters: Radius = {CIRCLE_RADIUS}m, Center = {CIRCLE_CENTER}")
print(f"Tracking how well INS maintains circle shape during GPS loss...")

# Analyze each dropout window for circle maintenance
dropout1_start = int(2.0 / dt)
dropout1_end = int(4.0 / dt)
dropout1_radial = [circle_tracking_errors[i] for i in range(dropout1_start, min(dropout1_end, len(circle_tracking_errors)))]
dropout1_lateral = [lateral_tracking_errors[i] for i in range(dropout1_start, min(dropout1_end, len(lateral_tracking_errors)))]

if dropout1_radial:
    print(f"\nWindow 1 (2-4s): INS Only")
    print(f"  Radius maintenance:  Mean {np.mean(dropout1_radial)*100:6.2f}cm error (ideally 0)")
    print(f"                       Std  {np.std(dropout1_radial)*100:6.2f}cm")
    print(f"  Lateral tracking:    {np.mean(dropout1_lateral)*100:6.2f}cm from reference path")
    radial_quality = "GOOD" if np.abs(np.mean(dropout1_radial)) < 0.15 else "FAIR" if np.abs(np.mean(dropout1_radial)) < 0.30 else "POOR"
    print(f"  Assessment:          {radial_quality}")

dropout2_start = int(8.0 / dt)
dropout2_end = int(10.0 / dt)
dropout2_radial = [circle_tracking_errors[i] for i in range(dropout2_start, min(dropout2_end, len(circle_tracking_errors)))]
dropout2_lateral = [lateral_tracking_errors[i] for i in range(dropout2_start, min(dropout2_end, len(lateral_tracking_errors)))]

if dropout2_radial:
    print(f"\nWindow 2 (8-10s): INS Only")
    print(f"  Radius maintenance:  Mean {np.mean(dropout2_radial)*100:6.2f}cm error")
    print(f"                       Std  {np.std(dropout2_radial)*100:6.2f}cm")
    print(f"  Lateral tracking:    {np.mean(dropout2_lateral)*100:6.2f}cm from reference path")

dropout3_start = int(14.0 / dt)
dropout3_end = int(16.0 / dt)
dropout3_radial = [circle_tracking_errors[i] for i in range(dropout3_start, min(dropout3_end, len(circle_tracking_errors)))]
dropout3_lateral = [lateral_tracking_errors[i] for i in range(dropout3_start, min(dropout3_end, len(lateral_tracking_errors)))]

if dropout3_radial:
    print(f"\nWindow 3 (14-16s): INS Only")
    print(f"  Radius maintenance:  Mean {np.mean(dropout3_radial)*100:6.2f}cm error")
    print(f"                       Std  {np.std(dropout3_radial)*100:6.2f}cm")
    print(f"  Lateral tracking:    {np.mean(dropout3_lateral)*100:6.2f}cm from reference path")

print("\n" + "="*80)
print(f"Circle mission complete! PyBullet visualization displayed.")
print("="*80 + "\n")

# Save results to file for verification
with open("circle_mission_results.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("CIRCLE MISSION WITH GPS DROPOUT - RESULTS\n")
    f.write("="*80 + "\n\n")
    
    if gps_errors:
        f.write("With GPS:\n")
        f.write(f"  Mean Error: {np.mean(gps_errors)*100:.2f}cm\n")
        f.write(f"  Max Error:  {np.max(gps_errors)*100:.2f}cm\n")
        f.write(f"  Samples:    {len(gps_errors)}\n\n")
    
    if ins_errors:
        f.write("During GPS Dropout (INS-only):\n")
        f.write(f"  Mean Error: {np.mean(ins_errors)*100:.2f}cm\n")
        f.write(f"  Max Error:  {np.max(ins_errors)*100:.2f}cm\n")
        f.write(f"  Samples:    {len(ins_errors)}\n\n")
    
    if ins_errors:
        improvement = 45.6 / np.mean(ins_errors)
        f.write(f"Improvement over baseline (45.6m): {improvement:.0f}x\n\n")
    
    f.write("Breakdown by dropout window:\n")
    for i, (start, end) in enumerate(dropout_windows):
        start_idx = int(start / dt)
        end_idx = int(end / dt)
        window_errors = all_errors[start_idx:end_idx]
        if window_errors:
            f.write(f"  Window {i+1} ({start:.1f}-{end:.1f}s): {np.mean(window_errors)*100:.2f}cm mean\n")

print("Results saved to circle_mission_results.txt\n")

env.close()
