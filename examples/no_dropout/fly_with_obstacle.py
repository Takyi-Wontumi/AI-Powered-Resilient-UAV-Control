"""
Fly a drone around an obstacle in PyBullet.

This example:
- Creates a DroneMissionEnv with AttitudeRate control
- Spawns a box obstacle in 3D space
- Takes off and then follows a path around/near the obstacle
- Checks for contact with the obstacle

Run:
    python examples/fly_with_obstacle.py

Example custom run:
    python examples/fly_with_obstacle.py --trajectory circle --radius 1.2 --period 14 --obstacle-x 0.0 --obstacle-y 0.0 --obstacle-z 1.0
"""

import argparse
import os
import sys
import time
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
from AI_UAV_Tests.core.trajectories_library import Trajectories as T


def thrust_to_action(u1, mass, g=9.81):
    """Convert thrust [N] to normalized AttitudeRate throttle in [-1, 1]."""
    hover_t = mass * g
    a0 = (u1 / hover_t - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


def safe_disconnect_env(env):
    if env is None:
        return
    try:
        if hasattr(env, "bc") and env.bc is not None:
            env.bc.disconnect()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass


def add_box_obstacle(env, center_xyz, size_xyz, color_rgba):
    """Spawn a static box obstacle and return body id."""
    hx, hy, hz = 0.5 * np.asarray(size_xyz, dtype=float)
    collision_id = env.bc.createCollisionShape(
        env.bc.GEOM_BOX,
        halfExtents=[float(hx), float(hy), float(hz)],
    )
    visual_id = env.bc.createVisualShape(
        env.bc.GEOM_BOX,
        halfExtents=[float(hx), float(hy), float(hz)],
        rgbaColor=[float(c) for c in color_rgba],
    )
    obstacle_id = env.bc.createMultiBody(
        baseMass=0.0,  # static obstacle
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])],
        baseOrientation=env.bc.getQuaternionFromEuler([0.0, 0.0, 0.0]),
    )
    env.bc.addUserDebugText(
        "Obstacle",
        [float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2] + hz + 0.1)],
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.2,
        lifeTime=0.0,
    )
    return obstacle_id


def get_reference(t, args):
    """Build reference position/velocity with center offset around obstacle."""
    if args.trajectory == "circle":
        omega = 2.0 * np.pi / args.period
        x = args.obstacle_x + args.radius * np.cos(omega * t)
        y = args.obstacle_y + args.radius * np.sin(omega * t)
        z = args.flight_z
        vx = -args.radius * omega * np.sin(omega * t)
        vy = args.radius * omega * np.cos(omega * t)
        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float)

    if args.trajectory == "square":
        pos, vel = T.square_traj(t, side=args.side, period=args.period, z=args.flight_z)
        pos = pos.astype(float)
        vel = vel.astype(float)
        pos[0] += args.obstacle_x - 0.5 * args.side
        pos[1] += args.obstacle_y - 0.5 * args.side
        return pos, vel

    if args.trajectory == "point":
        pos = np.array([args.obstacle_x + args.radius, args.obstacle_y, args.flight_z], dtype=float)
        return pos, np.zeros(3, dtype=float)

    # hover
    return np.array([0.0, 0.0, args.flight_z], dtype=float), np.zeros(3, dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fly mission with a spawned box obstacle.",
    )
    parser.add_argument("--duration", type=float, default=25.0, help="Total simulation time [s].")
    parser.add_argument("--speedup", type=float, default=1.0, help="Time scale factor. >1 runs faster.")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=["circle", "square", "point", "hover"],
        help="Reference trajectory after takeoff.",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Circle/point offset radius [m].")
    parser.add_argument("--side", type=float, default=1.2, help="Square side length [m].")
    parser.add_argument("--period", type=float, default=12.0, help="Trajectory period [s].")
    parser.add_argument("--flight-z", type=float, default=1.0, help="Reference altitude [m].")
    parser.add_argument("--takeoff-z", type=float, default=0.25, help="Initial takeoff target [m].")
    parser.add_argument("--takeoff-trigger-z", type=float, default=0.18, help="Enable path when drone exceeds this z [m].")
    parser.add_argument("--xy-speed-limit", type=float, default=1.0, help="Clamp reference XY speed [m/s].")

    parser.add_argument("--obstacle-x", type=float, default=1.0)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-z", type=float, default=1.0)
    parser.add_argument("--obstacle-size-x", type=float, default=0.5)
    parser.add_argument("--obstacle-size-y", type=float, default=0.5)
    parser.add_argument("--obstacle-size-z", type=float, default=0.5)
    parser.add_argument("--obstacle-color", nargs=4, type=float, default=[1.0, 0.2, 0.2, 1.0], help="RGBA in [0,1].")

    parser.add_argument("--trail", default=True, action=argparse.BooleanOptionalAction, help="Draw drone trail.")
    parser.add_argument("--debug-every", type=int, default=100, help="Print status every N steps.")
    parser.add_argument("--render", default=True, action=argparse.BooleanOptionalAction, help="Use PyBullet GUI.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0.")
    if args.period <= 0.0:
        raise ValueError("--period must be > 0.")
    if args.xy_speed_limit <= 0.0:
        raise ValueError("--xy-speed-limit must be > 0.")

    obstacle_size = np.array([args.obstacle_size_x, args.obstacle_size_y, args.obstacle_size_z], dtype=float)
    obstacle_center = np.array([args.obstacle_x, args.obstacle_y, args.obstacle_z], dtype=float)
    safety_radius = 0.6 * max(obstacle_size[0], obstacle_size[1])
    if args.trajectory == "circle" and args.radius <= safety_radius:
        print(
            f"[warning] radius={args.radius:.2f} is close to obstacle footprint. "
            f"Consider radius > {safety_radius:.2f} m."
        )

    env = None
    try:
        env = DroneMissionEnv(
            physics="PyBulletPhysics",
            control_mode="AttitudeRate",
            drone_model="cf21x_bullet",
            dropout_mode="NONE",
            render_mode="human" if args.render else None,
        )
        env.drone.control = AttitudeRate(
            bc=env.bc,
            drone=env.drone,
            time_step=env.TIME_STEP,
        )

        env.reset()
        obstacle_id = add_box_obstacle(env, obstacle_center, obstacle_size, args.obstacle_color)
        print(
            "Spawned obstacle:",
            f"center={obstacle_center.tolist()}",
            f"size={obstacle_size.tolist()}",
        )

        quad = QuadcopterPID(dt=env.TIME_STEP)
        quad.reset()

        env.set_target(np.array([0.0, 0.0, args.takeoff_z], dtype=float))
        path_active = False
        prev_pos = None

        dt = env.TIME_STEP
        steps = int(args.duration / dt)
        print(f"Starting obstacle mission for {steps} steps.")

        for k in range(steps):
            if args.render:
                time.sleep(max(0.0, dt / max(args.speedup, 1e-6)))

            env.mission_time += dt
            t = env.mission_time

            x = env.drone.xyz
            v = env.drone.xyz_dot
            ang = env.drone.rpy
            rate = env.drone.rpy_dot

            if (not path_active) and (x[2] > args.takeoff_trigger_z):
                path_active = True
                print("Switching to path tracking.")

            if path_active:
                pos_ref, vel_ref = get_reference(t, args)
                xy_speed = np.linalg.norm(vel_ref[:2])
                if xy_speed > args.xy_speed_limit:
                    vel_ref[:2] *= args.xy_speed_limit / max(xy_speed, 1e-9)
                env.set_target(pos_ref)
            else:
                pos_ref = env.get_mission_reference()
                vel_ref = np.zeros(3, dtype=float)

            quad.inject_external_state(x, v, ang, rate)
            z_ref = env.get_mission_reference()[2]
            ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)

            action = np.zeros(4, dtype=np.float32)
            action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
            action[1:4] = np.clip(np.asarray(ctrl["rates_des"], dtype=np.float32) / (np.pi / 3.0), -1.0, 1.0)

            _, _, terminated, truncated, _ = env.step(action)

            if args.trail and prev_pos is not None:
                env.bc.addUserDebugLine(
                    prev_pos.tolist(),
                    np.asarray(x, dtype=float).tolist(),
                    lineColorRGB=[0.05, 0.8, 1.0],
                    lineWidth=1.8,
                    lifeTime=0.0,
                )
            prev_pos = np.asarray(x, dtype=float).copy()

            contacts = env.bc.getContactPoints(bodyA=env.drone.body_unique_id, bodyB=obstacle_id)
            if len(contacts) > 0:
                print(f"Collision with obstacle at t={t:.2f}s. Ending run.")
                break

            if args.debug_every > 0 and (k % args.debug_every == 0):
                print(
                    f"t={t:6.2f}s z={x[2]:5.3f} speed={np.linalg.norm(v):5.3f} "
                    f"path={path_active} pos=({x[0]:+.2f},{x[1]:+.2f},{x[2]:+.2f})"
                )

            if terminated or truncated:
                print("Environment terminated due to safety condition.")
                break

        print("Finished obstacle mission.")
    finally:
        safe_disconnect_env(env)


if __name__ == "__main__":
    main()
