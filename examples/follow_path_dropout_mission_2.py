"""
MissionEnv test WITH PATH + KEYBOARD-ACTIVATED DROPOUT

Keys:
    H → Toggle Hover-at-current-position (HOV)
    R → Toggle Return-to-Home (RTH)
    C → Clear dropout (resume mission)

GOAL:
- Ground start
- Gentle takeoff
- Path following
- Manual dropout injection
- NO reset inside loop
"""

import sys, os
import time
import numpy as np

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.trajectories_library import Trajectories as path


# =========================================================
# Helper: thrust (N) → AttitudeRate action[0]
# =========================================================
def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


# =========================================================
# MAIN
# =========================================================
def main():

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------
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

    obs, info = env.reset()   # ONE TIME ONLY

    # -----------------------------------------------------
    # Controller
    # -----------------------------------------------------
    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    # -----------------------------------------------------
    # Mission setup
    # -----------------------------------------------------
    TAKEOFF_Z = 0.16
    PATHFOLLOW_TRIGGER_Z = 0.15

    env.set_target(np.array([0.0, 0.0, TAKEOFF_Z]))
    path_active = False

    dt = env.TIME_STEP
    T_final = 20.0
    steps = int(T_final / dt)

    print("Mission started.")
    print("Press H (hover), R (RTH), C (clear dropout)")

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    for k in range(steps):
        time.sleep(dt)

        # Advance mission time
        env.mission_time += dt
        t = env.mission_time

        # -------------------------------------------------
        # Keyboard handling (EDGE-TRIGGERED)
        # -------------------------------------------------
        keys = env.bc.getKeyboardEvents()

        if ord('h') in keys and keys[ord('h')] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT: HOVER")
            env.dropout_mgr.mode = "HOV"
            env.trigger_dropout()

        if ord('r') in keys and keys[ord('r')] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT: RETURN TO HOME")
            env.dropout_mgr.mode = "RTH"
            env.trigger_dropout()

        if ord('l') in keys and keys[ord('l')] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT: LAND")
            env.dropout_mgr.mode = "LAND"
            env.trigger_dropout()

        if ord('c') in keys and keys[ord('c')] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT CLEARED")
            env.clear_dropout()

        # -------------------------------------------------
        # Read state
        # -------------------------------------------------
        x = env.drone.xyz
        v = env.drone.xyz_dot
        ang = env.drone.rpy
        rate = env.drone.rpy_dot

        # -------------------------------------------------
        # Activate path AFTER liftoff
        # -------------------------------------------------
        if not path_active and x[2] > PATHFOLLOW_TRIGGER_Z:
            print(">> Switching to path following")
            path_active = True

        if path_active and not env.dropout_mgr.active:
            pos_ref, vel_ref = path.circle_traj(t)
            env.set_target(pos_ref)
        else:
            pos_ref = env.get_mission_reference()
            vel_ref = np.zeros(3)

        # -------------------------------------------------
        # Controller
        # -------------------------------------------------
        quad.inject_external_state(x, v, ang, rate)

        z_ref = env.get_mission_reference()[2]

        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)
        rates_des = ctrl["rates_des"]
        U1 = ctrl["thrust_cmd"]

        # Build AttitudeRate action
        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(U1, quad.m, quad.g)
        action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

        # Step physics
        obs, reward, done, truncated, info = env.step(action)

        # Debug
        if k % 100 == 0:
            print(
                f"t={t:.2f}s  z={x[2]:.3f}  "
                f"path={path_active}  dropout={env.dropout_mgr.active}"
            )

        if done:
            print("Mission terminated due to safety violation.")
            break

    env.close()
    print("Mission finished.")


if __name__ == "__main__":
    main()
