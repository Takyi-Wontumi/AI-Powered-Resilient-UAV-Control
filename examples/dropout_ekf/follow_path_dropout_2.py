"""
Phoenix Trajectory Following with QuadcopterPID Controller

- Phoenix: Hover environment + AttitudeRate inner loop + full PyBullet physics
- QuadcopterPID: position → attitude → rate PID cascade (Crazyflie style)
- Trajectories: provides reference pos/vel (circle, square, etc.)

Action to AttitudeRate:
    a[0] = thrust command in [-1, 1]
    a[1] = roll_rate command  (normalized)
    a[2] = pitch_rate command (normalized)
    a[3] = yaw_rate command   (normalized)

AttitudeRate internally does:
    rpy_dot_target = a[1:4] * (π / 3)      # rad/s (≈ ±60 deg/s)
"""
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import time
import numpy as np
import matplotlib.pyplot as plt

from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.control import AttitudeRate

# Your controller + buffer
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID, StateBuffer

# Trajectories
from AI_UAV_Tests.core.trajectories_library import Trajectories as path


# =========================================================
#  Helper: map thrust (N) → AttitudeRate normalized action[0]
# =========================================================
def thrust_to_action(U1: float, mass: float, g: float = 9.81) -> float:
    """
    Convert physical thrust U1 (N) from QuadcopterPID to normalized throttle [-1, 1] for AttitudeRate.

    Assumes QuadcopterPID clamps U1 to [0.5 mg, 1.3 mg].
    Map:
        0.5 mg → -1
        1.3 mg → +1
    """
    hover_T = mass * g
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


# =========================================================
#  Dropout schedule
# =========================================================
class DropoutSchedule:
    def __init__(self, start_time=6.0, duration=3.0, mode="hold"):
        """
        start_time : when dropout begins [s]
        duration   : how long dropout lasts [s]
        mode       : "hold" | "noise"
            - "hold": dead-reckon using last x,v (StateBuffer.predict)
            - "noise": corrupt measurements with Gaussian noise
        """
        self.start = float(start_time)
        self.end = float(start_time + duration)
        self.mode = str(mode)

    def active(self, t: float) -> bool:
        return self.start <= t <= self.end


# =========================================================
#  main
# =========================================================
def main():
    env: DroneHoverBulletEnv = DroneHoverBulletEnv(render_mode="human")

    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP
    )

    env.enable_reset_distribution = True
    env.domain_randomization = -1.0

    quad = QuadcopterPID(dt=env.TIME_STEP)
    state_buf = StateBuffer()

    dropout = DropoutSchedule(start_time=6.0, duration=3.0, mode="hold")

    obs, info = env.reset()
    quad.reset()
    t = 0.0

    T_final = 20.0
    dt = env.TIME_STEP
    steps = int(T_final / dt)
    print("steps:", steps)

    # Logs
    log_t = []
    log_pos = []
    log_ref = []
    log_pwm = []
    log_thrust = []
    log_dropout = []   # True/False per step

    for k in range(steps):
        time.sleep(dt)

        # --- raw measurements from Phoenix ---
        x_meas = env.drone.xyz
        v_meas = env.drone.xyz_dot
        ang = env.drone.rpy
        rate = env.drone.rpy_dot

        # --- dropout-aware state selection ---
        dropout_on = dropout.active(t)

        if not dropout_on:
            # normal sensing
            x = x_meas
            v = v_meas
            state_buf.update(x, v, t)
        else:
            # dropout: no fresh position/velocity
            if dropout.mode == "hold":
                # dead-reckon from last known x,v
                x, v = state_buf.predict(t)
            elif dropout.mode == "noise":
                # degrade measurements (still "available" but corrupted)
                x = x_meas + np.random.normal(0.0, 0.05, size=3)
                v = v_meas + np.random.normal(0.0, 0.05, size=3)
            else:
                raise ValueError(f"Unknown dropout.mode: {dropout.mode}")

        # Inject degraded/estimated state into controller
        quad.inject_external_state(x, v, ang, rate)

        # Reference trajectory
      #   pos_ref, vel_ref = path.point_traj((0, 0, 1))
        pos_ref, vel_ref = path.circle_traj(t, z=1.5)
        z_ref = pos_ref[2]

        # During dropout, don't chase velocity reference (reduces blow-up / windup)
        if dropout_on:
            vel_ref = np.zeros(3)

        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)

        rates_des = ctrl["rates_des"]      # rad/s
        U1 = ctrl["thrust_cmd"]            # N

        # Build AttitudeRate action
        action = np.zeros(4, dtype=np.float32)

        action[0] = thrust_to_action(U1, mass=quad.m, g=quad.g)
        rate_norm = rates_des / (np.pi / 3.0)
        action[1:4] = np.clip(rate_norm, -1.0, 1.0)

        # Step Phoenix
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Logs (log the state you actually fed the controller)
        log_t.append(t)
        log_pos.append(x.copy())
        log_ref.append(pos_ref.copy())
        log_pwm.append(float(action[0]))
        log_thrust.append(float(U1))
        log_dropout.append(dropout_on)

        t += dt

        if done:
            quad.reset()
            obs, info = env.reset()
            t = 0.0
            # IMPORTANT: buffer is now stale; reset it so dropout doesn’t use garbage
            state_buf = StateBuffer()

    # ---- plots ----
    log_t = np.array(log_t)
    log_pos = np.vstack(log_pos)
    log_ref = np.vstack(log_ref)
    log_dropout = np.array(log_dropout, dtype=bool)

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]"]

    for i in range(3):
        axs[i].plot(log_t, log_pos[:, i], label=f"{labels[i]} actual/used")
        axs[i].plot(log_t, log_ref[:, i], "--", label=f"{labels[i]} ref")
        axs[i].grid(True)
        axs[i].legend()

        # Shade dropout window (makes the plot honest)
        if log_dropout.any():
            axs[i].fill_between(
                log_t,
                axs[i].get_ylim()[0],
                axs[i].get_ylim()[1],
                where=log_dropout,
                alpha=0.15
            )

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
