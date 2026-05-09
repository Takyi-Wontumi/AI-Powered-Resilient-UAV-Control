"""
Use an Attitude-Rate Controller to stabilize the drone for Take-off task.

Author: Sven Gronauer
Modified by: Lawrence Wontumi (2025)
"""

import time
import numpy as np

from phoenix_drone_simulation.envs.takeoff import DroneTakeOffBulletEnv
from phoenix_drone_simulation.envs.control import AttitudeRate


def main():
    # === ENVIRONMENT ===
    env = DroneTakeOffBulletEnv(render_mode="human")

    # Use Attitude-Rate PID instead of raw PWM
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP
    )

    env.enable_reset_distribution = True
    env.domain_randomization = 0.0   # make takeoff behavior predictable

    dt = env.TIME_STEP
    SIM_STEPS = 20000                # <<< increase simulation duration here

    # === ACTION FORMAT ===
    # a = [thrust_norm, roll_rate_norm, pitch_rate_norm, yaw_rate_norm]
    actions = np.zeros((SIM_STEPS, 4))

    # Use hover + slight extra thrust for takeoff
    actions[:, 0] = env.drone.HOVER_ACTION + 0.10
    actions[:, 1] = 0.0
    actions[:, 2] = 0.0
    actions[:, 3] = 0.0

    obs, info = env.reset()

    for step in range(SIM_STEPS):

        obs, reward, terminated, truncated, info = env.step(actions[step])

        # Current altitude (z)
        z = env.drone.xyz[2]

        if step % 100 == 0:
            print(f"Step {step:05d} | z = {z:.3f} m")

        if terminated or truncated:
            print("Reset triggered. Restarting episode.\n")
            obs, info = env.reset()

        time.sleep(dt)


if __name__ == "__main__":
    main()
