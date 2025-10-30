"""
Simple flight demo: move the drone using the PositionController only.
No training, no reinforcement learning — just PID control.
"""

import numpy as np
import time
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.position import PositionController


def goto(env, pos_ctrl, target, tolerance=0.05, max_steps=4000):
    """
    Move the drone to a target (x, y, z) using position control.
    Stops once within tolerance of the target.
    """
    for step in range(max_steps):
        # === Read drone state ===
        pos = np.array(env.drone.position())
        vel = np.array(env.drone.linear_velocity())

        # === Compute control command ===
        pwm = pos_ctrl.act(desired_position=target)

        # === Apply to simulator ===
        obs, reward, terminated, truncated, info = env.step(pwm)

        # === Distance from goal ===
        error = np.linalg.norm(target - pos)

        # === Display progress ===
        if step % 100 == 0:
            print(f"[{step:04d}] pos={pos.round(2)}, error={error:.3f} m")

        # === End if goal reached ===
        if error < tolerance:
            print(f"✅ Reached target {target} with error {error:.3f} m")
            break

        # === Optional real-time pacing ===
        time.sleep(env.TIME_STEP)

        if terminated or truncated:
            print("Environment terminated early.")
            break


def main():
    # === Create environment ===
    env = DroneHoverBulletEnv(render_mode="human")

    # === Create controller ===
    pos_ctrl = PositionController(drone=env.drone, bc=env.bc, time_step=env.TIME_STEP)
    pos_ctrl.reset()

    # === Reset environment ===
    obs, info = env.reset()

    # === Sequence of waypoints ===
    waypoints = [
        np.array([0.0, 0.0, 1.0]),   # take off and hover
        np.array([0.5, 0.0, 1.0]),   # move forward
        np.array([0.5, 0.5, 1.0]),   # move right
        np.array([0.0, 0.0, 1.0])    # return to center
    ]

    for wp in waypoints:
        print(f"\nGoing to waypoint {wp}")
        goto(env, pos_ctrl, wp)

    print("🏁 Mission complete — closing environment.")
    env.close()


if __name__ == "__main__":
    main()
