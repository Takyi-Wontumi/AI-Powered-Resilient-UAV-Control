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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
import time
import numpy as np
import matplotlib.pyplot as plt

### from the envs
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.control import AttitudeRate

#adding custom quadcopter
from AI_UAV_Tests.quadcopter_env import QuadcopterPID

# Your trajectory library
from AI_UAV_Tests.trajectories_library import Trajectories as path


# =========================================================
#  Helper: map thrust (N) → AttitudeRate normalized action[0]
# =========================================================
def thrust_to_action(U1: float, mass: float, g: float = 9.81) -> float:
    """
    This takes physical thrust U1 (N) from QuadcopterPID and convert
    to normalized throttle in [-1, 1] for AttitudeRate.

    Note: U1 is clamped in QuadcopterPID to [0.5 mg, 1.3 mg].
    We map:
        0.5 mg  → -1
        1.3 mg  → +1
    """
    hover_T = mass * g
    # Normalize relative to hover: 0.5mg → -1, 1.3mg → +1
    a0 = (U1 / hover_T - 0.9) / 0.4
    return float(np.clip(a0, -1.0, 1.0))


# =========================================================
#  main section with for the simulation
# =========================================================
def main():
    env: DroneHoverBulletEnv = DroneHoverBulletEnv(render_mode="human")

    # controlling the drone with AttitudeRate rather than PWM
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP
    )

    # make reset deterministic or add reduce randomness to the drone
    env.enable_reset_distribution = True
    env.domain_randomization = -1.0

    quad = QuadcopterPID(dt=env.TIME_STEP)

    obs, info = env.reset()
    t = 0.0

    T_final = 20.0                   # this is the total time for the simulation
    dt = env.TIME_STEP
    
    steps = int(T_final / dt)
    print(steps)

    # input("\nPress ENTER to start the simulation...")

    # Logs for plotting
    log_t = []
    log_pos = []
    log_ref = []
    log_pwm = []
    log_thrust = []

    for k in range(steps):
        # Real-time pacing
        time.sleep(dt)

        # Getting current drone state from Phoenix
        x = env.drone.xyz
        v = env.drone.xyz_dot
        ang = env.drone.rpy
        rate = env.drone.rpy_dot

        # Inject into QuadcopterPID so it uses real physics state
        quad.inject_external_state(x, v, ang, rate)

        # Get reference from trajectory
        pos_ref, vel_ref = path.point_traj((0, 0, 7))
        z_ref = pos_ref[2]
        
        # full PID with thrust commands
        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)

        rates_des = ctrl["rates_des"]     # [p_des, q_des, r_des] rad/s
        U1        = ctrl["thrust_cmd"]    # thrust (N)

        log_thrust.append(U1)       #logging thrust

        
        # Build AttitudeRate action
        
        action = np.zeros(4, dtype=np.float32)

        # a[0]: thrust command
        action[0] = thrust_to_action(U1, mass=quad.m, g=quad.g)
        log_pwm.append(action[0])       #logging PWM
        print(f"Thrust: {U1} N, PWM: {action[0]}")

        # a[1:4]: normalized attitude rate commands
        # AttitudeRate: rpy_dot_target = a[1:4] * (π/3)
        rate_norm = rates_des / (np.pi / 3.0)
        action[1:4] = np.clip(rate_norm, -1.0, 1.0)

        
        # Setting up the  Phoenix environment
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Log time, state, and reference
        log_t.append(t)
        log_pos.append(x.copy())
        log_ref.append(pos_ref.copy())

        t += dt

        if done:
            # Reset controller integrals & env
            quad.reset()
            obs, info = env.reset()
            t = 0.0

    # 4. Convert logs to arrays and plot tracking

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
