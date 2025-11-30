"""
Baseline Attitude-mode PD-controlled flight demo for DroneFollowPathEnv.
Author: Lawrence Wontumi (2025)

- Pure attitude/thrust-norm control loop
- PD + optional I on z-axis
- Includes thrust-slew limiter
- Logs and plots tracking errors
"""

import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
#  Repo path setup
# =========================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phoenix_drone_simulation.envs.base import DroneBaseEnv
from phoenix_drone_simulation.envs.followpath_fixed import DroneFollowPathEnv
from AI_UAV_Tests.trajectories_library import Trajectories as path


# =========================================================
#  Controller parameters
# =========================================================
MAX_TILT_RAD = np.deg2rad(10)
ANGLE_SCALE  = np.pi / 9

# base normalized parameters (Attitude mode)
BASE_THRUST_NORM  = 0.13           # close to neutral hover
Kp_xy, Kd_xy      = 0.6, 0.4
Kp_z,  Kd_z, Ki_z = 0.5, 0.4, 0.02  # damped z response for hover
Z_INT_LIMIT       = 0.2
Z_THRUST_SCALE    = 0.004           # smaller response to z error
THRUST_SLEW       = 0.002          # smoother thrust transition


prev_thrust = BASE_THRUST_NORM
z_err_int = 0.0


def reset_controller_state(base):
    global prev_thrust, z_err_int
    prev_thrust = base
    z_err_int = 0.0


# =========================================================
#  PD Controller
# =========================================================
def tracking_controller(env: DroneBaseEnv, pos_ref: np.ndarray, vel_ref: np.ndarray) -> np.ndarray:
    """Maps position/velocity error to [thrust_norm, roll, pitch, yaw_rate]."""
    global prev_thrust, z_err_int

    pos_err = pos_ref - env.drone.xyz
    vel_err = vel_ref - env.drone.xyz_dot
    dt = getattr(env, "TIME_STEP", 1.0 / env.SIM_FREQ)

    # --- Lateral PD ---
    acc_xy = Kp_xy * pos_err[:2] + Kd_xy * vel_err[:2]
    desired_roll  = np.clip(acc_xy[1] / env.G, -MAX_TILT_RAD, MAX_TILT_RAD)
    desired_pitch = np.clip(-acc_xy[0] / env.G, -MAX_TILT_RAD, MAX_TILT_RAD)
    roll_cmd  = desired_roll / ANGLE_SCALE
    pitch_cmd = desired_pitch / ANGLE_SCALE

    # --- Vertical PID ---
    z_err_int = np.clip(z_err_int + pos_err[2] * dt, -Z_INT_LIMIT, Z_INT_LIMIT)
    acc_z = Kp_z * pos_err[2] + Kd_z * vel_err[2] + Ki_z * z_err_int

    # --- Attitude thrust scaling ---
    base = BASE_THRUST_NORM
    scale = Z_THRUST_SCALE
    limit = (0.0, 1.0)
    slew = THRUST_SLEW

    raw_thrust = np.clip(base + acc_z * scale, *limit)

    # Slew limiting
    thrust_cmd = np.clip(
        prev_thrust + np.clip(raw_thrust - prev_thrust, -slew, slew),
        *limit
    )
    prev_thrust = thrust_cmd

    if np.random.rand() < 0.02:
        print(f"[Attitude] thrust={thrust_cmd:.4f}, z={env.drone.xyz[2]:.3f} m")

    return np.array([thrust_cmd, roll_cmd, pitch_cmd, 0.0], dtype=np.float32)


# =========================================================
#  Simulation
# =========================================================
if __name__ == "__main__":
    CONTROL_MODE = "Attitude"
    traj_fn = path.hover_traj

    env: DroneBaseEnv = DroneFollowPathEnv(
        trajectory_fn=traj_fn,
        control_mode=CONTROL_MODE,
        render_mode="human",
        done_dist_threshold=5.0,
    )

    base = BASE_THRUST_NORM
    obs, info = env.reset(seed=42)
    env.enable_reset_distribution = False
    reset_controller_state(base)

    # print(f"Environment initialized in {CONTROL_MODE} mode. Press [Enter] to start simulation.")
    # input()

    SIM_DURATION = 25.0
    LOOP_SLEEP = env.TIME_STEP
    error_log = []

    print("Starting PD baseline simulation...")
    t0 = time.time()
    step = 0

    while True:
        t_sim = env.iteration / env.SIM_FREQ
        pos_ref, vel_ref = traj_fn(t_sim)
        action = tracking_controller(env, pos_ref, vel_ref)
        obs, reward, terminated, truncated, info = env.step(action)

        pos_err = pos_ref - env.drone.xyz
        err_norm = np.linalg.norm(pos_err)
        error_log.append([t_sim, *pos_err, err_norm])

        if step % 100 == 0:
            pos = env.drone.xyz
            print(f"Step {step:04d} | Pos={pos} | Thrust={prev_thrust:.4f} | Err={err_norm:.3f} m")

        if terminated or truncated:
            print("Simulation ended early.")
            break
        if t_sim >= SIM_DURATION:
            print("Simulation duration reached.")
            break

        step += 1
        time.sleep(LOOP_SLEEP)

    wall_time = time.time() - t0
    print(f"Simulation wall time: {wall_time:.2f} s")
    env.close()

    # =========================================================
    #  Error Plot + Summary
    # =========================================================
    df = pd.DataFrame(error_log, columns=["t", "ex", "ey", "ez", "err_norm"])
    rms_error = np.sqrt(np.mean(df["err_norm"] ** 2))
    max_error = np.max(df["err_norm"])

    print("\n--- Tracking Performance ---")
    print(f"RMS Error: {rms_error:.3f} m")
    print(f"Max Error: {max_error:.3f} m")

    plt.figure(figsize=(8, 4))
    plt.plot(df["t"], df["err_norm"], label="Tracking Error [m]")
    plt.axhline(0.1, color="r", linestyle="--", label="10 cm Spec")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.title(f"PD Hover Tracking ({CONTROL_MODE} mode, auto-scaled)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
