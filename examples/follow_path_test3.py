import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
#  Repo path setup
# =========================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
follow_path_test3.py
--------------------

Trajectory following using Webots-style full PID controller integrated
into the Phoenix Drone Simulation environment.

This script:
 - Loads a Phoenix drone environment
 - Replaces the default controller with WebotsFullPID
 - Pulls a reference trajectory (circle, square, etc)
 - Tracks each waypoint using vel_xy + alt + vz control
 - Sends PWM commands (4 motors) to env.step()
"""

import time
import numpy as np

# Phoenix imports
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.followpath_fixed import DroneFollowPathEnv

# Your full Webots PID wrapper
from phoenix_drone_simulation.envs.test_controller import WebotsFullPID

# Your trajectory library
from AI_UAV_Tests.trajectories_library import Trajectories

def disable_latency(drone):
    """Disable Phoenix's built-in latency system so that the controller
    always receives the full command (5 elements)."""

    drone.use_latency = False
    drone.latency = 0.0
    drone.buf_size = 1
    drone.action_buffer = np.zeros((1, drone.act_dim))


def main():

    # =========================================================
    # 1. Load environment
    # =========================================================
    env = DroneFollowPathEnv(
        render_mode="human",
        trajectory="circle",
        debug=False,
        latency=0.0,                  # Disable latency in constructor
    )

    env.enable_reset_distribution = False
    env.domain_randomization = -1.0

    # =========================================================
    # 2. Replace Phoenix controller with Webots full PID
    # =========================================================
    env.drone.control = WebotsFullPID(
        env.drone, env.bc, env.TIME_STEP
    )

    # MUST disable latency buffering so the controller receives
    # your 5-element command instead of a corrupted 4-element buffer.
    disable_latency(env.drone)

    # =========================================================
    # 3. Reset and simulation loop setup
    # =========================================================
    obs, info = env.reset()
    dt = env.TIME_STEP
    T = 6000  # simulation steps

    print("Starting trajectory tracking with Webots PID...")

    # =========================================================
    # 4. Main control loop
    # =========================================================
    for step in range(T):

        # ------------------------------------------------------
        # a) Get reference waypoint from environment
        # ------------------------------------------------------
        desired_pos = np.array(env.target_pos)  # [x, y, z]

        x, y, z = env.drone.xyz
        vx, vy, vz = env.drone.xyz_dot

        # ------------------------------------------------------
        # b) XY velocity command toward the target (simple P-control)
        # ------------------------------------------------------
        vel_gain = 1.0

        desired_vx = vel_gain * (desired_pos[0] - x)
        desired_vy = vel_gain * (desired_pos[1] - y)

        desired_vx = np.clip(desired_vx, -1.0, 1.0)
        desired_vy = np.clip(desired_vy, -1.0, 1.0)

        # altitude + vertical velocity
        desired_alt = desired_pos[2]
        desired_vz = 0.0

        # yaw control disabled for now
        desired_yaw_rate = 0.0

        # ------------------------------------------------------
        # c) Build the Webots command vector (5 values)
        # ------------------------------------------------------
        command = np.array([
            desired_vx,        # desired vx
            desired_vy,        # desired vy
            desired_yaw_rate,  # yaw rate
            desired_alt,       # target altitude
            desired_vz,        # desired vertical velocity
        ], dtype=np.float32)

        # ------------------------------------------------------
        # d) Controller takes 5 inputs → outputs 4 PWMs
        # ------------------------------------------------------
        pwm = env.drone.control.act(command)

        # ------------------------------------------------------
        # e) Send 4-dimensional motor command to Phoenix
        # ------------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(pwm)

        if step % 100 == 0:
            print(f"[{step:04d}] "
                  f"Target=({desired_pos[0]:.2f}, {desired_pos[1]:.2f}, {desired_pos[2]:.2f}) "
                  f"| Pos=({x:.2f}, {y:.2f}, {z:.2f})")

        if terminated or truncated:
            print("Episode ended — resetting.")
            obs, info = env.reset()

        time.sleep(dt)

    env.close()
    print("Finished trajectory tracking.")


if __name__ == "__main__":
    main()
