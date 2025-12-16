"""
Phoenix Trajectory Following with QuadcopterPID Controller + 60 FPS MP4 Export
"""

import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pb
import imageio.v3 as iio      # MP4 writer with ffmpeg backend

### Phoenix imports
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.control import AttitudeRate

### Custom controller
from AI_UAV_Tests.quadcopter_env import QuadcopterPID

### Trajectory
from AI_UAV_Tests.trajectories_library import Trajectories as path


# =========================================================
#  Helper: thrust → normalized command
# =========================================================
def thrust_to_action(U1: float, mass: float, g: float = 9.81) -> float:
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


# =========================================================
#  MAIN
# =========================================================
def main():

    env: DroneHoverBulletEnv = DroneHoverBulletEnv(render_mode="human")

    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP
    )

    env.enable_reset_distribution = True
    env.domain_randomization = -5.0

    quad = QuadcopterPID(dt=env.TIME_STEP)

    obs, info = env.reset()
    t = 0.0

    T_final = 1.0
    dt = env.TIME_STEP
    steps = int(T_final / dt)

    log_t, log_pos, log_ref = [], [], []

    # Frame buffers
    frames = []
    frame_times = []

    print(f"[INFO] Running simulation for {steps} steps...")

    for k in range(steps):

        # ===== Phoenix → controller state injection =====
        x = env.drone.xyz
        v = env.drone.xyz_dot
        ang = env.drone.rpy
        rate = env.drone.rpy_dot
        quad.inject_external_state(x, v, ang, rate)

        # ===== Trajectory reference =====
        pos_ref, vel_ref = path.circle_traj(t)
        z_ref = pos_ref[2]

        # ===== Controller output =====
        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)
        rates_des = ctrl["rates_des"]
        U1 = ctrl["thrust_cmd"]

        # ===== Build AttitudeRate action =====
        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(U1, mass=quad.m, g=quad.g)
        rate_norm = rates_des / (np.pi / 3.0)
        action[1:4] = np.clip(rate_norm, -1.0, 1.0)

        # ===== Step Phoenix =====
        obs, reward, terminated, truncated, info = env.step(action)

        # ===== Logging =====
        log_t.append(t)
        log_pos.append(x.copy())
        log_ref.append(pos_ref.copy())

        # ===== High-speed camera capture =====
        if k == 0:
            start_time = time.time()

        width, height, rgb, depth, seg = pb.getCameraImage(
            width=1280,
            height=720,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_img = np.reshape(rgb, (height, width, 4))[:, :, :3]
        frames.append(rgb_img)
        frame_times.append(time.time() - start_time)

        # ===== Simulation time update =====
        t += dt

        if terminated or truncated:
            quad.reset()
            obs, info = env.reset()
            t = 0.0

    # =========================================================
    # 60 FPS MP4 EXPORT
    # =========================================================
    print("[INFO] Preparing MP4 output...")

    frame_times = np.array(frame_times)
    frame_times -= frame_times[0]

    target_fps = 60
    target_dt = 1.0 / target_fps

    max_time = frame_times[-1]
    uniform_times = np.arange(0, max_time, target_dt)

    # Interpolate frame indices
    interp_idx = np.interp(
        uniform_times,
        frame_times,
        np.arange(len(frames))
    ).astype(int)

    smooth_frames = [frames[i] for i in interp_idx]

    print("[INFO] Writing MP4 video (60 FPS)...")

    iio.imwrite(
        "simulation_60fps.mp4",
        np.stack(smooth_frames),
        fps=60,
        codec="libx264",
        quality=8
    )

    print("[INFO] Saved → simulation_60fps.mp4")

    # =========================================================
    # Plot tracking
    # =========================================================
    log_t = np.array(log_t)
    log_pos = np.vstack(log_pos)
    log_ref = np.vstack(log_ref)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]"]

    for i in range(3):
        axs[i].plot(log_t, log_pos[:, i], label=f"{labels[i]} actual")
        axs[i].plot(log_t, log_ref[:, i], "--", label=f"{labels[i]} ref")
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
