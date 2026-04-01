"""
MissionEnv test WITH FlightMission + keyboard-activated dropout.

Keys:
    H -> Toggle Hover-at-current-position (HOV)
    R -> Toggle Return-to-Home (RTH)
    L -> Toggle Land (LAND)
    C -> Clear dropout (resume mission)
"""

import sys, os
import time
import argparse
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
from AI_UAV_Tests.trajectories_library import FlightMission


# =========================================================
# Helper: thrust (N) -> AttitudeRate action[0]
# =========================================================
def thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


def termination_reason(env: DroneMissionEnv) -> str:
    xyz = env.drone.xyz
    rpy = env.drone.rpy
    rpy_dot = env.drone.rpy_dot

    if env.dropout_mgr.active and env.dropout_mgr.mode == "LAND" and xyz[2] <= env.ground_z:
        return f"landing complete (z={xyz[2]:.3f} <= ground_z={env.ground_z:.3f})"
    if env.airborne and xyz[2] < env.ground_z:
        return f"ground impact after liftoff (z={xyz[2]:.3f} < ground_z={env.ground_z:.3f})"
    if (np.abs(rpy[:2]) > np.deg2rad(60.0)).any():
        return f"attitude limit exceeded (roll/pitch > 60 deg): rpy={np.rad2deg(rpy)} deg"
    if (np.abs(rpy_dot) > np.deg2rad(300.0)).any():
        return f"angular-rate limit exceeded (>300 deg/s): rpy_dot={np.rad2deg(rpy_dot)} deg/s"
    return "terminated by environment"


def run_physics_preflight(
    mission_plan: FlightMission,
    speedup: float,
    xy_speed_limit: float,
    min_z_ref: float,
):
    """
    Physics-based preflight preview using the same stack as the real mission:
    PyBullet + AttitudeRate + QuadcopterPID.
    """
    env = DroneMissionEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human",
    )
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP,
    )
    obs, info = env.reset()
    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    env.set_target(mission_plan(0.0)[0])

    dt = env.TIME_STEP
    steps = int(mission_plan.total_time / dt)

    table_header = (
        f"{'step':>6}  {'task':<10}  {'task_time(s)':>12}  {'mission_time(s)':>15}  "
        f"{'speed(m/s)':>10}  {'x(m)':>8}  {'y(m)':>8}  {'z(m)':>8}  {'mode':<10}"
    )
    print("Physics preflight started.")
    print(table_header)
    print("-" * len(table_header))

    prev_pos = None
    for k in range(steps):
        time.sleep(max(0.0, dt / max(speedup, 1e-6)))
        env.mission_time += dt
        t = env.mission_time

        x = env.drone.xyz
        v = env.drone.xyz_dot
        ang = env.drone.rpy
        rate = env.drone.rpy_dot

        phase_name = mission_plan.phase_name_at(t)
        pos_ref, vel_ref = mission_plan(t)
        pos_ref = np.asarray(pos_ref, dtype=float).copy()
        vel_ref = np.asarray(vel_ref, dtype=float).copy()

        if phase_name != "landing":
            pos_ref[2] = max(float(pos_ref[2]), float(min_z_ref))
        xy_norm = float(np.linalg.norm(vel_ref[:2]))
        if xy_norm > float(xy_speed_limit):
            vel_ref[:2] *= float(xy_speed_limit) / max(xy_norm, 1e-9)

        env.set_target(pos_ref)

        quad.inject_external_state(x, v, ang, rate)
        z_ref = env.get_mission_reference()[2]
        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)
        rates_des = ctrl["rates_des"]
        U1 = ctrl["thrust_cmd"]

        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(U1, quad.m, quad.g)
        action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)

        if prev_pos is not None:
            line_color = mission_plan.trail_color_at_time(t=t, dropout_active=False)
            env.bc.addUserDebugLine(
                prev_pos.tolist(),
                x.tolist(),
                lineColorRGB=line_color,
                lineWidth=2.0,
                lifeTime=0.0,
            )
        prev_pos = x.copy()

        if k % 10 == 0:
            _, phase_elapsed, phase_total = mission_plan.phase_info_at(t)
            print(
                f"{k:6d}  {phase_name:<10}  {phase_elapsed:5.2f}/{phase_total:5.2f}  "
                f"{t:6.2f}/{mission_plan.total_time:6.2f}  {np.linalg.norm(v):10.3f}  "
                f"{x[0]:8.3f}  {x[1]:8.3f}  {x[2]:8.3f}  {'physics':<10}"
            )

        if done:
            print(f"Physics preflight terminated: {termination_reason(env)}")
            env.close()
            return False

    print("Physics preflight completed.")
    env.close()
    return True


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Mission follow + dropout + preflight from example script")
    parser.add_argument(
        "--preflight",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable preflight check before main mission run",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Run only the preflight preview and exit")
    parser.add_argument(
        "--preflight-speed",
        type=float,
        default=5.0,
        help="Preflight animation speed multiplier (1x to 5x)",
    )
    parser.add_argument(
        "--preflight-mode",
        choices=["physics", "kinematic"],
        default="physics",
        help="Use physics preflight (realistic) or kinematic preflight (analytic)",
    )
    parser.add_argument(
        "--preflight-dashboard",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show/hide the matplotlib preflight dashboard window",
    )
    parser.add_argument("--preflight-hz", type=float, default=50.0, help="Preflight sampling frequency [Hz]")
    parser.add_argument("--xy-speed-limit", type=float, default=1.0, help="Limit reference XY speed [m/s]")
    parser.add_argument(
        "--min-z-ref",
        type=float,
        default=0.25,
        help="Minimum z reference outside landing [m]",
    )
    args = parser.parse_args()
    if not (1.0 <= args.preflight_speed <= 5.0):
        parser.error("--preflight-speed must be between 1 and 5.")
    if args.xy_speed_limit <= 0.0:
        parser.error("--xy-speed-limit must be > 0.")
    if args.min_z_ref < 0.0:
        parser.error("--min-z-ref must be >= 0.")
    if args.preflight_only and not args.preflight:
        parser.error("--preflight-only requires preflight to be enabled. Use --preflight.")

    # -----------------------------------------------------
    # Mission plan (built once)
    # -----------------------------------------------------
    mission_plan = FlightMission(default_z=1.0, ground_z=0.0)
    mission_plan.add_takeoff(duration=3.0, target_z=1.0)
    mission_plan.add_circle(duration=12.0, radius=1.0, period=12.0, z=1.0, center_xy=(-1.0, 0.0))  # 1 full loop
    mission_plan.add_hover(duration=2.0, z=1.0)
    mission_plan.add_point(duration=2.0, target=(-0.5, -0.5, 1.0))
    mission_plan.add_square(duration=12.0, side=1.0, period=12.0, z=1.0, offset_xy=(-0.5, -0.5))  # 1 full loop
    mission_plan.add_hover(duration=2.0, z=1.0)
    mission_plan.add_landing(duration=4.0, ground_z=0.075)
    mission_plan.add_hover(duration=1.0, z=0.075)
    mission_plan.add_takeoff(duration=12.0, target_z=5.0)
    mission_plan.add_hover(duration=3.0, z=5.0)
    mission_plan.add_landing(duration=12.0, ground_z=0.055)

    if args.preflight:
        print(f"Running {args.preflight_mode} preflight check at {args.preflight_speed:.1f}x speed...")
        if args.preflight_dashboard:
            print("Showing preflight dashboard window...")
            mission_plan.preflight_check(
                speedup=args.preflight_speed,
                sample_hz=args.preflight_hz,
                block=True,
            )

        if args.preflight_mode == "physics":
            run_physics_preflight(
                mission_plan=mission_plan,
                speedup=args.preflight_speed,
                xy_speed_limit=args.xy_speed_limit,
                min_z_ref=args.min_z_ref,
            )
        elif not args.preflight_dashboard:
            print("Kinematic preflight selected with --no-preflight-dashboard; skipping preflight window.")
    if args.preflight_only:
        print("Preflight-only mode complete.")
        return

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------
    env = DroneMissionEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human",
    )

    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP,
    )

    obs, info = env.reset()  # ONE TIME ONLY

    # -----------------------------------------------------
    # Controller
    # -----------------------------------------------------
    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    # Keep mission env target in sync from start.
    env.set_target(mission_plan(0.0)[0])

    dt = env.TIME_STEP
    T_final = mission_plan.total_time
    steps = int(T_final / dt)

    print("Mission started.")
    print("Press H (hover), R (RTH), L (land), C (clear dropout)")
    print("Mission summary:", mission_plan.summary())
    table_header = (
        f"{'step':>6}  {'task':<10}  {'task_time(s)':>12}  {'mission_time(s)':>15}  "
        f"{'speed(m/s)':>10}  {'x(m)':>8}  {'y(m)':>8}  {'z(m)':>8}  {'dropout':<8}"
    )
    print(table_header)
    print("-" * len(table_header))

    # Logs for end-of-flight stability plots
    log_t = []
    log_pos = []
    log_ref = []
    log_speed = []
    log_dropout = []
    prev_pos = None

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    for k in range(steps):
        time.sleep(dt)

        # Advance mission time
        env.mission_time += dt
        t = env.mission_time

        # -------------------------------------------------
        # Keyboard handling (edge-triggered)
        # -------------------------------------------------
        keys = env.bc.getKeyboardEvents()

        if ord("h") in keys and keys[ord("h")] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT: HOVER")
            env.dropout_mgr.mode = "HOV"
            env.trigger_dropout()

        if ord("r") in keys and keys[ord("r")] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT: RETURN TO HOME")
            env.dropout_mgr.mode = "RTH"
            env.trigger_dropout()

        if ord("l") in keys and keys[ord("l")] & env.bc.KEY_WAS_TRIGGERED:
            print(">> DROPOUT: LAND")
            env.dropout_mgr.mode = "LAND"
            env.trigger_dropout()

        if ord("c") in keys and keys[ord("c")] & env.bc.KEY_WAS_TRIGGERED:
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
        # Reference selection
        # -------------------------------------------------
        current_phase = mission_plan.phase_name_at(t)
        if not env.dropout_mgr.active:
            pos_ref, vel_ref = mission_plan(t)
            pos_ref = np.asarray(pos_ref, dtype=float).copy()
            vel_ref = np.asarray(vel_ref, dtype=float).copy()

            # Keep altitude reference away from ground except during mission landing.
            if current_phase != "landing":
                pos_ref[2] = max(float(pos_ref[2]), float(args.min_z_ref))

            # Cap lateral reference speed to reduce aggressive tilt and altitude loss.
            xy_norm = float(np.linalg.norm(vel_ref[:2]))
            if xy_norm > float(args.xy_speed_limit):
                vel_ref[:2] *= float(args.xy_speed_limit) / max(xy_norm, 1e-9)

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

        # Colored trail (phase-colored, dropout shown in yellow)
        if prev_pos is not None:
            line_color = mission_plan.trail_color_at_time(
                t=t,
                dropout_active=env.dropout_mgr.active,
            )
            env.bc.addUserDebugLine(
                prev_pos.tolist(),
                x.tolist(),
                lineColorRGB=line_color,
                lineWidth=2.0,
                lifeTime=0.0,
            )
        prev_pos = x.copy()

        # Logs
        log_t.append(float(t))
        log_pos.append(x.copy())
        log_ref.append(pos_ref.copy())
        log_speed.append(float(np.linalg.norm(v)))
        log_dropout.append(bool(env.dropout_mgr.active))

        # Dashboard (table rows)
        if k % 10 == 0:
            phase_name, phase_elapsed, phase_total = mission_plan.phase_info_at(t)
            dropout_state = env.dropout_mgr.mode if env.dropout_mgr.active else "NONE"
            print(
                f"{k:6d}  {phase_name:<10}  {phase_elapsed:5.2f}/{phase_total:5.2f}  "
                f"{t:6.2f}/{T_final:6.2f}  {np.linalg.norm(v):10.3f}  "
                f"{x[0]:8.3f}  {x[1]:8.3f}  {x[2]:8.3f}  {dropout_state:<8}"
            )
            if (k // 10) % 25 == 24:
                print("-" * len(table_header))

        if done:
            print(f"Mission terminated: {termination_reason(env)}")
            break

    env.close()

    # End-of-run stability plots (XYZ vs time)
    if len(log_t) > 0:
        t_arr = np.asarray(log_t)
        pos_arr = np.vstack(log_pos)
        ref_arr = np.vstack(log_ref)
        dropout_arr = np.asarray(log_dropout, dtype=bool)

        fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        labels = ["X [m]", "Y [m]", "Z [m]"]
        for i in range(3):
            axs[i].plot(t_arr, pos_arr[:, i], label=f"{labels[i]} actual")
            axs[i].plot(t_arr, ref_arr[:, i], "--", label=f"{labels[i]} ref")
            if np.any(dropout_arr):
                axs[i].fill_between(
                    t_arr,
                    axs[i].get_ylim()[0],
                    axs[i].get_ylim()[1],
                    where=dropout_arr,
                    alpha=0.12,
                    label="dropout" if i == 0 else None,
                )
            axs[i].grid(True)
            axs[i].legend(loc="best")

        axs[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.show()

    print("Mission finished.")


if __name__ == "__main__":
    main()
