"""
MissionEnv path following with manual noise dropout.

Keys:
    N -> Toggle measurement noise (dropout) on/off
    H -> Hover at current position (HOV)
    R -> Return to home (RTH)
    L -> Land (LAND)
    C -> Clear dropout (resume mission)

Notes:
- Uses DroneMissionEnv (ground start, continuous mission)
- Uses QuadcopterPID with external state injection
- Noise is applied to position/velocity measurements only
"""

import sys, os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

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

    # -----------------------------------------------------
    # Noise setup
    # -----------------------------------------------------
    NOISE_STD_POS = 0.15
    NOISE_STD_VEL = 0.15

    # -----------------------------------------------------
    # Scenario schedule
    # -----------------------------------------------------
    SCENARIOS = [
        {"name": "normal", "duration": 10.0, "noise": False, "dropout": "NONE"},
        {"name": "noise", "duration": 10.0, "noise": True, "dropout": "NONE"},
        {"name": "dropout", "duration": 10.0, "noise": False, "dropout": "HOV"},
    ]
    RESET_BETWEEN_SCENARIOS = True
    DROPOUT_START_T = 4.0
    DROPOUT_DURATION = None  # None = keep dropout active until scenario ends

    dt = env.TIME_STEP
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("Mission started (auto sequence: normal -> noise -> dropout).")

    def run_scenario(s):
        if RESET_BETWEEN_SCENARIOS:
            env.reset()
            quad.reset()
        env.set_target(np.array([0.0, 0.0, TAKEOFF_Z]))
        path_active = False
        dropout_triggered = False
        dropout_cleared = False

        steps = int(s["duration"] / dt)
        rows = []
        for k in range(steps):
            time.sleep(dt)
            env.mission_time += dt
            t = env.mission_time

            # -------------------------------------------------
            # Read state (raw)
            # -------------------------------------------------
            x_meas = env.drone.xyz
            v_meas = env.drone.xyz_dot
            ang = env.drone.rpy
            rate = env.drone.rpy_dot

            # Apply noisy measurements to controller input
            if s["noise"]:
                x_used = x_meas + np.random.normal(0.0, NOISE_STD_POS, size=3)
                v_used = v_meas + np.random.normal(0.0, NOISE_STD_VEL, size=3)
            else:
                x_used = x_meas
                v_used = v_meas

            # -------------------------------------------------
            # Activate path AFTER liftoff (use true altitude)
            # -------------------------------------------------
            if not path_active and x_meas[2] > PATHFOLLOW_TRIGGER_Z:
                path_active = True

            if path_active and not env.dropout_mgr.active:
                pos_ref, vel_ref = path.circle_traj(t)
                env.set_target(pos_ref)
            else:
                pos_ref = env.get_mission_reference()
                vel_ref = np.zeros(3)

            # -------------------------------------------------
            # Dropout logic
            # -------------------------------------------------
            if s["dropout"] != "NONE" and (t >= DROPOUT_START_T) and not dropout_triggered:
                env.dropout_mgr.mode = s["dropout"]
                env.trigger_dropout()
                dropout_triggered = True

            if DROPOUT_DURATION is not None and dropout_triggered and not dropout_cleared:
                if t >= DROPOUT_START_T + DROPOUT_DURATION:
                    env.clear_dropout()
                    dropout_cleared = True

            # -------------------------------------------------
            # Controller
            # -------------------------------------------------
            quad.inject_external_state(x_used, v_used, ang, rate)

            z_ref = env.get_mission_reference()[2]

            ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)
            rates_des = ctrl["rates_des"]
            U1 = ctrl["thrust_cmd"]

            action = np.zeros(4, dtype=np.float32)
            action[0] = thrust_to_action(U1, quad.m, quad.g)
            action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

            obs, reward, done, truncated, info = env.step(action)

            pos_err = pos_ref - x_meas if pos_ref is not None else np.zeros(3)
            err_norm = float(np.linalg.norm(pos_err))

            rows.append({
                "scenario": s["name"],
                "step": k,
                "t": t,
                "x": x_meas[0], "y": x_meas[1], "z": x_meas[2],
                "vx": v_meas[0], "vy": v_meas[1], "vz": v_meas[2],
                "x_used": x_used[0], "y_used": x_used[1], "z_used": x_used[2],
                "vx_used": v_used[0], "vy_used": v_used[1], "vz_used": v_used[2],
                "roll": ang[0], "pitch": ang[1], "yaw": ang[2],
                "p": rate[0], "q": rate[1], "r": rate[2],
                "ref_x": pos_ref[0], "ref_y": pos_ref[1], "ref_z": pos_ref[2],
                "err_x": pos_err[0], "err_y": pos_err[1], "err_z": pos_err[2],
                "err_norm": err_norm,
                "thrust_cmd": float(U1),
                "rates_des_x": rates_des[0], "rates_des_y": rates_des[1], "rates_des_z": rates_des[2],
                "action0": action[0], "action1": action[1], "action2": action[2], "action3": action[3],
                "noise_active": s["noise"],
                "dropout_active": env.dropout_mgr.active,
                "dropout_mode": env.dropout_mgr.mode,
                "reward": float(reward),
            })

            if k % 100 == 0:
                print(
                    f"[{s['name']}] step={k} z={x_meas[2]:.3f} "
                    f"path={path_active} noise={s['noise']} dropout={env.dropout_mgr.active}"
                )

            if done:
                print(f"[{s['name']}] Mission terminated due to safety violation.")
                break

        return rows

    all_logs = {}
    for s in SCENARIOS:
        print(f">> Running scenario: {s['name']}")
        rows = run_scenario(s)
        all_logs[s["name"]] = rows

        if rows:
            csv_path = os.path.join(
                log_dir, f"follow_path_dropout_noise_{s['name']}.csv"
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {csv_path}")

    env.close()
    print("Mission finished.")

    # -----------------------------------------------------
    # Plot overlay: x, y, z and error for each scenario
    # -----------------------------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]", "Position Error [m]"]

    for name, rows in all_logs.items():
        if not rows:
            continue
        t = np.array([r["t"] for r in rows])
        x = np.array([r["x"] for r in rows])
        y = np.array([r["y"] for r in rows])
        z = np.array([r["z"] for r in rows])
        e = np.array([r["err_norm"] for r in rows])

        axs[0].plot(t, x, label=name)
        axs[1].plot(t, y, label=name)
        axs[2].plot(t, z, label=name)
        axs[3].plot(t, e, label=name)

    for i in range(4):
        axs[i].grid(True)
        axs[i].set_ylabel(labels[i])
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
