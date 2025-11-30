"""
Stable hover using PositionController + Attitude inner loop.
"""

import time
import numpy as np

from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.control import Attitude
from phoenix_drone_simulation.envs.position import PositionController


def main():
    # === Environment ===
    env = DroneHoverBulletEnv(render_mode="human")
    env.enable_reset_distribution = False
    env.domain_randomization = -1

    # === Replace default control with cascaded Position → Attitude ===
    pos_ctrl = PositionController(
        drone=env.drone,
        bc=env.bc,
        time_step=env.TIME_STEP,
    )

    obs, info = env.reset()
    target = np.array([0, 0, 1.0])      # hover point

    dt = env.TIME_STEP

    for step in range(5000):
        # Compute attitude PWM (pos → att → motor)
        pwm = pos_ctrl.act(target)

        # Convert PWM values back to action [-1,1] range
        # because env.step expects NORMALIZED INPUT
        action = (pwm - 30000) / 30000
        action = np.clip(action, 0.1, 0.1)

        obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(dt)


if __name__ == "__main__":
    main()
