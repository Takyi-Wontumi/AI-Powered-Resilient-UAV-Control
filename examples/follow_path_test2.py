import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
#  Repo path setup
# =========================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv

env = DroneHoverBulletEnv(control_mode="Attitude", render_mode="human")
obs, info = env.reset(seed=1)

print("\n=== Hover Thrust Sweep Test ===")
print("Goal: find thrust command ([-1,1]) that results in ~zero vertical velocity.\n")

thrust_values = np.linspace(-1.0, 1.0, 21)

for u in thrust_values:
    print(f"\nTesting thrust={u:.2f} ...")
    for _ in range(150):  # ~0.75 seconds at 200 Hz
        action = np.array([u, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        z = env.drone.xyz[2]
        vz = env.drone.xyz_dot[2]
        print(f"  z={z:.3f} m   vz={vz:.3f} m/s", end="\r")
        time.sleep(env.TIME_STEP)

env.close()
print("\n=== Finished ===")
