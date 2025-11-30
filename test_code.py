"""
Debugged Crazyflie-style Attitude controller for Phoenix.

Step 1: HOVER ONLY (no circle)
Step 2: Turn on CIRCLE once hover is stable.
"""

import time
import numpy as np

from phoenix_drone_simulation.envs.trajectory import DroneFollowTrajectoryEnv
from AI_UAV_Tests.trajectories_library import Trajectories


# ============================================================
#  Controller
# ============================================================
class AttitudeHoverCircleController:
    def __init__(self, hover_cmd=0.0):
        """
        hover_cmd: normalized thrust command (action[0]) s.t.
                   PWM = 45000 + 10000 * hover_cmd
        Try values in [-0.2, 0.2]. We'll print behavior and tune.
        """
        # Z gains (start modest)
        self.Kp_z = 1.5
        self.Kd_z = 0.8

        self.hover_cmd = hover_cmd
        self.g = 9.81
        self.max_angle = np.deg2rad(10.0)  # we will keep attitude small

    def act_hover_only(self, state, pos_ref, vel_ref, dt):
        """
        Pure hover: only altitude loop, no XY motion.
        """
        pos = state["pos"]
        vel = state["vel"]

        z = pos[2]
        vz = vel[2]
        z_ref = pos_ref[2]

        ez = z_ref - z
        evz = -vz

        u = self.hover_cmd + self.Kp_z * ez + self.Kd_z * evz
        u = float(np.clip(u, -0.128, 0.128*2))

        # roll, pitch, yaw commands = 0  (we just want stability)
        roll_cmd = 0.0
        pitch_cmd = 0.0
        yaw_cmd = 0.0

        action = np.array([u, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
        return action

    def act_circle(self, state, pos_ref, vel_ref, dt):
        """
        Circle: same Z control, plus VERY gentle XY tilt.
        """
        pos = state["pos"]
        vel = state["vel"]
        rpy = state["rpy"]

        # ---------- Z control (same as hover) ----------
        z = pos[2]
        vz = vel[2]
        z_ref = pos_ref[2]

        ez = z_ref - z
        evz = -vz

        u = self.hover_cmd + self.Kp_z * ez + self.Kd_z * evz
        u = float(np.clip(u, -0.6, 0.6))

        # ---------- XY velocity-based tilt (weak) ----------
        psi = rpy[2]
        c, s = np.cos(psi), np.sin(psi)

        # global -> body frame velocities
        vx_b =  vel[0] * c + vel[1] * s
        vy_b = -vel[0] * s + vel[1] * c

        vx_ref_b =  vel_ref[0] * c + vel_ref[1] * s
        vy_ref_b = -vel_ref[0] * s + vel_ref[1] * c

        ex = vx_ref_b - vx_b
        ey = vy_ref_b - vy_b

        # tiny XY gains so we don't fling it
        Kp_xy = np.array([0.8, 0.8])
        ax_des = Kp_xy[0] * ex
        ay_des = Kp_xy[1] * ey

        # desired roll (phi), pitch (theta)
        phi_des   =  ay_des / self.g
        theta_des = -ax_des / self.g

        phi_des   = float(np.clip(phi_des, -self.max_angle, self.max_angle))
        theta_des = float(np.clip(theta_des, -self.max_angle, self.max_angle))

        # NOTE: Attitude.act does:
        #   rpy_target = action[1:4] * (pi/18)
        # so action[1] = 1 => 10 deg
        # We want phi_des, so:
        roll_cmd  = phi_des / (np.pi / 18.0)
        pitch_cmd = theta_des / (np.pi / 18.0)
        yaw_cmd = 0.0

        roll_cmd  = float(np.clip(roll_cmd, -1.0, 1.0))
        pitch_cmd = float(np.clip(pitch_cmd, -1.0, 1.0))

        action = np.array([u, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32)
        return action


# ============================================================
#  Main
# ============================================================
def main():
    USE_CIRCLE = False  # <<< STEP 1: set False for hover debug

    # Trajectories
    def hover_traj(t):
        # position fixed at (0.4, 0, 1) because your circle starts at x=0.4
        pos = np.array([0.4, 0.0, 1.0], dtype=np.float32)
        vel = np.zeros(3, dtype=np.float32)
        return pos, vel

    def circ_traj(t):
        return Trajectories.circle_traj(
            t,
            radius=0.4,
            period=20.0,
            z=1.0
        )

    traj_fn = circ_traj if USE_CIRCLE else hover_traj

    env = DroneFollowTrajectoryEnv(
        trajectory_fn=traj_fn,
        control_mode="AttitudeRate",
        gui=True,
        render_mode="human",
    )
    env.enable_reset_distribution = False
    env.domain_randomization = -1

    # Start with hover_cmd = 0.0. If it still climbs, try -0.1, -0.2.
    ctrl = AttitudeHoverCircleController(hover_cmd=0.0)

    obs, info = env.reset()
    dt = env.TIME_STEP

    print("[INFO] Starting flight... USE_CIRCLE =", USE_CIRCLE)

    for step in range(6000):
        t = env.iteration / env.SIM_FREQ
        pos_ref, vel_ref = traj_fn(t)

        state = env.get_state()

        if USE_CIRCLE:
            action = ctrl.act_circle(state, pos_ref, vel_ref, dt)
        else:
            action = ctrl.act_hover_only(state, pos_ref, vel_ref, dt)

        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print("[FATAL] NaN/Inf in action, aborting. action =", action)
            break

        obs, rew, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            print(
                f"t={t:.2f} pos={state['pos']} ref={pos_ref} "
                f"vel={state['vel']} act={action}"
            )

        time.sleep(dt)


if __name__ == "__main__":
    main()
