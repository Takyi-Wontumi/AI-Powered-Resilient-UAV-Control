"""
Simple flight demo: move the drone using the PositionController only.
No training, no reinforcement learning — just PID control.
"""

import numpy as np
import time
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.position import PositionController

# === Initialize Environment ===
env = DroneHoverBulletEnv(render_mode="human")
env.bc.setRealTimeSimulation(0)  # manual stepping

# === Create Controller ===
pos_ctrl = PositionController(env.drone, env.bc, env.TIME_STEP)
obs, info = env.reset()

# === Let user see the scene first ===
print("PyBullet window is ready.")
input("Press [Enter] to start the simulation...")

# === Target position ===
target = np.array([0.0, 1.0, 1.0])   # move forward and hover

for step in range(5000):
    pwm = pos_ctrl.act(desired_position=target)
    obs, reward, terminated, truncated, info = env.step(pwm)
    pos = np.array(env.drone.xyz)

    if step % 100 == 0:
        print(f"{step:04d}: pos={pos.round(2)} m")

    # slow-motion playback
    time.sleep(env.TIME_STEP * 3)

    if terminated or truncated:
        print("Terminated early.")
        break

# === Post-flight pause ===
print("\n✅ Flight complete! Press [Enter] to close the simulation window...")
input()
env.close()
