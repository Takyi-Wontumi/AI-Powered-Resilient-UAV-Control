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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.followpath_dropout_mission import (
    DroneFollowPathDropoutMissionEnv,
)
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter
from AI_UAV_Tests.trajectories_library import FlightMission


def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


def build_mission():
    mission = FlightMission(default_z=1.0, ground_z=0.0)
    mission.add_takeoff(duration=3.0, target_z=1.0)
    mission.add_square(duration=12.0, side=1.0, period=12.0, z=1.0, offset_xy=(-0.5, -0.5))
    mission.add_hover(duration=2.0, z=1.0)
    mission.add_circle(duration=12.0, radius=0.75, period=12.0, z=1.0, center_xy=(-0.5, 0.0))
    mission.add_landing(duration=6.0, ground_z=0.055)
    return mission


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
    return parser.parse_args()


def main():
    args = parse_args()
    mission = build_mission()
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

    ekf = PhoenixEKFAdapter(
        dt=env.TIME_STEP,
    )
    ekf.reset(
        position=env.drone.xyz,
        velocity=env.drone.xyz_dot,
        attitude=env.drone.rpy,
        rates=env.drone.rpy_dot,
    )
    estimate = ekf.ekf.as_dict()

    print("Mission started with EKF state feedback.")
    if args.render:
        print("Keys: H=hover, R=return-home, L=land, C=clear dropout")
    else:
        print("Running headless. Use --render to enable the PyBullet GUI and keyboard dropout controls.")

    steps = int(mission.total_time / env.TIME_STEP)
    log_t = []
    log_ref = []
    log_measured = []
    for k in range(steps):
        time.sleep(env.TIME_STEP)

        if args.render:
            keys = env.bc.getKeyboardEvents()
            if ord("h") in keys and keys[ord("h")] & env.bc.KEY_WAS_TRIGGERED:
                env.dropout_mgr.mode = "HOV"
                env.trigger_dropout()
            if ord("r") in keys and keys[ord("r")] & env.bc.KEY_WAS_TRIGGERED:
                env.dropout_mgr.mode = "RTH"
                env.trigger_dropout()
            if ord("l") in keys and keys[ord("l")] & env.bc.KEY_WAS_TRIGGERED:
                env.dropout_mgr.mode = "LAND"
                env.trigger_dropout()
            if ord("c") in keys and keys[ord("c")] & env.bc.KEY_WAS_TRIGGERED:
                env.clear_dropout()

        if env.dropout_mgr.active:
            pos_ref = np.asarray(env.get_mission_reference(), dtype=float)
            vel_ref = np.zeros(3, dtype=float)
        else:
            pos_ref, vel_ref = env.current_reference()
            pos_ref = np.asarray(pos_ref, dtype=float)
            vel_ref = np.asarray(vel_ref, dtype=float)

        quad.inject_external_state(
            estimate["x"],
            estimate["v"],
            estimate["ang"],
            estimate["rate"],
        )

        z_ref = float(pos_ref[2])
        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)

        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)
        estimate = ekf.step(
            motor_omega=ctrl["omega_cmd"],
            position=env.drone.xyz,
            velocity=env.drone.xyz_dot,
            attitude=env.drone.rpy,
            rates=env.drone.rpy_dot,
            dropout_active=env.dropout_mgr.active,
            dt=env.TIME_STEP,
        )
        measured_pos = np.asarray(estimate["measurement"]["position"], dtype=float)
        log_t.append(float(env.mission_time))
        log_ref.append(pos_ref.copy())
        log_measured.append(measured_pos.copy())

        if k % 20 == 0:
            pos = estimate["x"]
            print(
                f"t={env.mission_time:6.2f}s  ref=({pos_ref[0]: .2f},{pos_ref[1]: .2f},{pos_ref[2]: .2f})  "
                f"est=({pos[0]: .2f},{pos[1]: .2f},{pos[2]: .2f})  "
                f"dropout={'ON' if env.dropout_mgr.active else 'OFF'}"
            )

        if done or truncated:
            print("Mission terminated.")
            break

    env.close()

    if log_t and not args.no_plot:
        t_arr = np.asarray(log_t, dtype=float)
        ref_arr = np.asarray(log_ref, dtype=float)
        measured_arr = np.asarray(log_measured, dtype=float)

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axis_labels = ["X [m]", "Y [m]", "Z [m]"]
        for i in range(3):
            axs[i].plot(t_arr, ref_arr[:, i], "--", linewidth=2.0, label="reference")
            axs[i].plot(t_arr, measured_arr[:, i], linewidth=1.5, label="measured")
            axs[i].set_ylabel(axis_labels[i])
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc="best")

        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("Reference vs Measured Position")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
