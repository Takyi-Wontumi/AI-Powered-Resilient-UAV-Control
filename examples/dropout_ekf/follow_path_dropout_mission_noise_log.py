"""
MissionEnv with path + manual dropout + manual noise, logging to CSV.

Keys:
    N -> Toggle measurement noise on/off
    H -> Hover at current position (HOV)
    R -> Return to home (RTH)
    L -> Land (LAND)
    C -> Clear dropout (resume mission)

CSV logs include:
    time, pos/vel (measured + used), rpy/rpy_dot, reference, action, thrust,
    rates, noise/dropout flags, reward, done.
"""

import sys, os
import time
import csv
import numpy as np

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.core.trajectories_library import Trajectories as path


# =========================================================
# Helper: thrust (N) -> AttitudeRate action[0]
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

    obs, info = env.reset()  # ONE TIME ONLY

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

    # -----------------------------------------------------
    # Noise setup
    # -----------------------------------------------------
    noise_active = False
    NOISE_STD_POS = 0.05
    NOISE_STD_VEL = 0.05

    dt = env.TIME_STEP
    T_final = 20.0
    steps = int(T_final / dt)

    # -----------------------------------------------------
    # CSV logging setup
    # -----------------------------------------------------
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    log_path = os.path.join(log_dir, f"mission_dropout_noise_{timestamp}.csv")

    header = [
        "step", "t",
        "noise_active", "noise_std_pos", "noise_std_vel",
        "dropout_active", "dropout_mode", "path_active",
        "x_meas", "y_meas", "z_meas",
        "vx_meas", "vy_meas", "vz_meas",
        "x_used", "y_used", "z_used",
        "vx_used", "vy_used", "vz_used",
        "roll", "pitch", "yaw",
        "p", "q", "r",
        "ref_x", "ref_y", "ref_z",
        "ref_vx", "ref_vy", "ref_vz",
        "action_0", "action_1", "action_2", "action_3",
        "U1", "rates_des_x", "rates_des_y", "rates_des_z",
        "reward", "done"
    ]

    print("Mission started.")
    print("Press N to toggle measurement noise.")
    print("Press H (hover), R (RTH), L (land), C (clear dropout).")
    print(f"Logging to: {log_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # -------------------------------------------------
        # MAIN LOOP
        # -------------------------------------------------
        for k in range(steps):
            time.sleep(dt)

            # Advance mission time
            env.mission_time += dt
            t = env.mission_time

            # -------------------------------------------------
            # Keyboard handling (EDGE-TRIGGERED)
            # -------------------------------------------------
            keys = env.bc.getKeyboardEvents()
            if ord('n') in keys and keys[ord('n')] & env.bc.KEY_WAS_TRIGGERED:
                noise_active = not noise_active
                state = "ON" if noise_active else "OFF"
                print(f">> NOISE DROPOUT {state}")

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
            # Read state (raw)
            # -------------------------------------------------
            x_meas = env.drone.xyz
            v_meas = env.drone.xyz_dot
            ang = env.drone.rpy
            rate = env.drone.rpy_dot

            # Apply noisy measurements to controller input
            if noise_active:
                x = x_meas + np.random.normal(0.0, NOISE_STD_POS, size=3)
                v = v_meas + np.random.normal(0.0, NOISE_STD_VEL, size=3)
            else:
                x = x_meas
                v = v_meas

            # -------------------------------------------------
            # Activate path AFTER liftoff (use true altitude)
            # -------------------------------------------------
            if not path_active and x_meas[2] > PATHFOLLOW_TRIGGER_Z:
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

            z_ref = env.get_mission_reference()[2] if env.get_mission_reference() is not None else pos_ref[2]
            ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)
            rates_des = ctrl["rates_des"]
            U1 = ctrl["thrust_cmd"]

            # Build AttitudeRate action
            action = np.zeros(4, dtype=np.float32)
            action[0] = thrust_to_action(U1, quad.m, quad.g)
            action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

            # Step physics
            obs, reward, done, truncated, info = env.step(action)

            # -------------------------------------------------
            # CSV row
            # -------------------------------------------------
            writer.writerow([
                k, t,
                int(noise_active), NOISE_STD_POS, NOISE_STD_VEL,
                int(env.dropout_mgr.active), env.dropout_mgr.mode, int(path_active),
                x_meas[0], x_meas[1], x_meas[2],
                v_meas[0], v_meas[1], v_meas[2],
                x[0], x[1], x[2],
                v[0], v[1], v[2],
                ang[0], ang[1], ang[2],
                rate[0], rate[1], rate[2],
                pos_ref[0], pos_ref[1], pos_ref[2],
                vel_ref[0], vel_ref[1], vel_ref[2],
                action[0], action[1], action[2], action[3],
                U1, rates_des[0], rates_des[1], rates_des[2],
                float(reward), int(done or truncated)
            ])

            # Debug
            if k % 100 == 0:
                print(
                    f"step={k}  z={x_meas[2]:.3f}  "
                    f"path={path_active}  noise={noise_active}  "
                    f"dropout={env.dropout_mgr.active}"
                )

            if done:
                print("Mission terminated due to safety violation.")
                break

    env.close()
    print("Mission finished.")


if __name__ == "__main__":
    main()
