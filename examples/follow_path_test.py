"""
Visualization test for DroneFollowPathEnv (fixed version)
Uses followpath_fixed and types via DroneBaseEnv.
"""

import numpy as np
import time

from phoenix_drone_simulation.envs.base import DroneBaseEnv
from phoenix_drone_simulation.envs.followpath_fixed import DroneFollowPathEnv
from AI_UAV_Tests.trajectories_library import Trajectories as path

# =========================================================
# Instantiate environment
# =========================================================
if __name__ == "__main__":
    # Type annotate as base env for clarity and Gym-style usage
    env: DroneBaseEnv = DroneFollowPathEnv(
        trajectory_fn=path.circle_traj,
        control_mode="PWM",
        render_mode="human",
    )

    obs, info = env.reset(seed=42)
    print("Environment initialized. Press [Enter] to start simulation.")
    input()

    total_steps = 2000
    t0 = time.time()

    for step in range(total_steps):
        # trivial proportional control example (random demo)
        action = env.action_space.sample()  # use RL or PID later
        obs, reward, terminated, truncated, info = env.step(action)

        # print position every 100 steps
        if step % 100 == 0:
            pos = env.drone.xyz
            print(f"Step {step:04d}: Position = {pos}")

        if terminated or truncated:
            print("Simulation ended early.")
            break

        # match real-time speed
        time.sleep(env.TIME_STEP)

    print(f"Simulation finished in {time.time() - t0:.2f} s")

    # Plot tracking error after run
    env.plot_error()
