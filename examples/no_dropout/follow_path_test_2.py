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

### from the envs
from phoenix_drone_simulation.envs.hover import DroneHoverBulletEnv
from phoenix_drone_simulation.envs.control import AttitudeRate

# Your controller + trajectory
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID

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
    # -------------------------------------------------
    # 1. Environment setup
    # -------------------------------------------------
    env: DroneHoverBulletEnv = DroneHoverBulletEnv(render_mode="human")

    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP
    )

    env.enable_reset_distribution = False
    env.domain_randomization = 0.0

    quad = QuadcopterPID(dt=env.TIME_STEP)

    obs, info = env.reset()
    quad.reset()

    print("action_space:", env.action_space.low, env.action_space.high)

    # -------------------------------------------------
    # 2. Mission Planner trajectory (THIS is the change)
    # -------------------------------------------------
    from AI_UAV_Tests.core.trajectories_library import MissionPlannerTrajectory

    MISSION_PATH = r"C:\Users\Lawrence Wontumi\Downloads\AI-Powered-Resilient-UAV-Control\Realworld_Deployment\Mission Planner\ChapelHill_Test2.mission"
    TOTAL_TIME = 60.0

    traj = MissionPlannerTrajectory(MISSION_PATH, total_time=TOTAL_TIME)

    # Explicit flight phases (no magic)
    traj.add_takeoff_min_jerk(t_start=0.0, duration=3.0, target_z=1.0)
    traj.add_hover(t_start=3.0, duration=2.0)
    traj.add_landing(t_start=55.0, duration=5.0, ground_z=0.0)

    print(traj.summary())

    # -------------------------------------------------
    # 3. Simulation timing
    # -------------------------------------------------
    dt = env.TIME_STEP
    steps = int((TOTAL_TIME + 10.0) / dt)  # extra time for landing
    t = 0.0

    # -------------------------------------------------
    # 4. Logs
    # -------------------------------------------------
    log_t = []
    log_pos = []
    log_ref = []
    log_thrust = []
    log_pwm = []

    # -------------------------------------------------
    # 5. Main control loop
    # -------------------------------------------------
    for k in range(steps):
        time.sleep(dt)

        # Phoenix state
        x = env.drone.xyz
        v = env.drone.xyz_dot
        ang = env.drone.rpy
        rate = env.drone.rpy_dot

        quad.inject_external_state(x, v, ang, rate)

        # ===== THIS IS THE MONEY LINE =====
        pos_ref, vel_ref = traj(t)
        z_ref = pos_ref[2]

        # Controller
        ctrl = quad.step(pos_ref, vel_ref, z_ref=z_ref)

        rates_des = ctrl["rates_des"]
        U1 = ctrl["thrust_cmd"]

        # Action
        action = np.zeros(4, dtype=np.float32)
        action[0] = thrust_to_action(U1, mass=quad.m, g=quad.g)
        action[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        # Logs
        log_t.append(t)
        log_pos.append(x.copy())
        log_ref.append(pos_ref.copy())
        log_thrust.append(U1)
        log_pwm.append(action[0])

        t += dt

        if terminated or truncated:
            print("Environment reset")
            quad.reset()
            obs, info = env.reset()
            t = 0.0

    # -------------------------------------------------
    # 6. Plot tracking
    # -------------------------------------------------
    log_t = np.array(log_t)
    log_pos = np.vstack(log_pos)
    log_ref = np.vstack(log_ref)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]"]

    for i in range(3):
        axs[i].plot(log_t, log_pos[:, i], label="actual")
        axs[i].plot(log_t, log_ref[:, i], "--", label="reference")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
