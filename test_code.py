

def reduction(value, order_sensitivity=0):
   import numpy as np
   return int (value * (order_sensitivity / 100))

print(reduction(45000, 1))
# """
# WORKING AttitudeRate Circle / Hover Demo for Phoenix Crazyflie
# Author: Lawrence Wontumi (2025)

# - Single PyBullet GUI (no double-GUI / ExampleBrowser conflict)
# - Stable altitude
# - Circle tracking using AttitudeRate
# """

# import os
# # Kill the stupid ExampleBrowser so Phoenix can own the GUI
# os.environ["PYBULLET_NO_EXAMPLE_BROWSER"] = "1"

# import time
# import numpy as np

# from phoenix_drone_simulation.envs.trajectory import DroneFollowTrajectoryEnv
# from AI_UAV_Tests.trajectories_library import Trajectories


# # ============================================================
# #   STABLE ATTITUDERATE CONTROLLER
# # ============================================================
# class StableAttRateController:
#     def __init__(self):
#         # Normalized thrust command (AttitudeRate: thrust_pwm = 30000 + 30000 * u)
#         # u ≈ 0.10–0.15 is around hover for this model
#         self.hover_cmd = 0.12

#         # Altitude PD
#         self.Kp_z = 6.0
#         self.Kd_z = 3.0

#         # XY: velocity error -> desired tilt (in body frame)
#         self.Kp_v = 1.6

#         # Attitude: angle error -> angle rate
#         self.Kp_ang = 6.0

#         self.g = 9.81
#         self.max_tilt = np.deg2rad(12.0)  # ±12° tilt limit

#     # ----------------- Altitude control -----------------
#     def thrust_ctrl(self, pos, vel, pos_ref):
#         z = pos[2]
#         vz = vel[2]
#         z_ref = pos_ref[2]

#         ez = z_ref - z
#         evz = -vz

#         u = self.hover_cmd + self.Kp_z * ez + self.Kd_z * evz
#         # Safe range for normalized thrust
#         u = float(np.clip(u, -0.2, 0.4))
#         return u

#     # ----------------- Main control law -----------------
#     def act(self, state, pos_ref, vel_ref, dt):
#         pos = state["pos"]
#         vel = state["vel"]
#         rpy = state["rpy"]
#         psi = rpy[2]

#         # ----- Z control -----
#         thrust_norm = self.thrust_ctrl(pos, vel, pos_ref)

#         # ----- XY velocity error in world frame -----
#         vel_err = vel_ref - vel  # [vx, vy, vz] world

#         # ----- Transform vel error to body frame -----
#         c = np.cos(psi)
#         s = np.sin(psi)
#         R_wb = np.array([[ c,  s],
#                          [-s,  c]])
#         vel_err_b = R_wb @ vel_err[:2]

#         # ----- Desired tilt from body-frame acceleration demand -----
#         ax_des = self.Kp_v * vel_err_b[0]   # forward/back
#         ay_des = self.Kp_v * vel_err_b[1]   # left/right

#         theta_des = -ax_des / self.g        # pitch
#         phi_des   =  ay_des / self.g        # roll

#         theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
#         phi_des   = np.clip(phi_des,   -self.max_tilt, self.max_tilt)

#         # ----- Angle error -> desired body rates -----
#         phi, theta = rpy[0], rpy[1]

#         e_phi   = phi_des   - phi
#         e_theta = theta_des - theta

#         p_des = self.Kp_ang * e_phi        # roll rate [rad/s]
#         q_des = self.Kp_ang * e_theta      # pitch rate [rad/s]
#         r_des = 0.0                        # no yaw spin by default

#         rate_scale = np.pi / 3.0           # AttitudeRate: action[1:4]*pi/3 = rpy_dot_target

#         roll_rate_norm  = float(np.clip(p_des / rate_scale,   -1.0, 1.0))
#         pitch_rate_norm = float(np.clip(q_des / rate_scale,   -1.0, 1.0))
#         yaw_rate_norm   = float(np.clip(r_des / rate_scale,   -1.0, 1.0))

#         action = np.array(
#             [thrust_norm, roll_rate_norm, pitch_rate_norm, yaw_rate_norm],
#             dtype=np.float32
#         )

#         if not np.isfinite(action).all():
#             print("[WARN] NaN in action, forcing hover:", action)
#             action = np.array([self.hover_cmd, 0.0, 0.0, 0.0], dtype=np.float32)

#         return action


# # ============================================================
# #   TRAJECTORIES
# # ============================================================
# def hover_traj(t):
#     pos = np.array([0.4, 0.0, 1.0], dtype=np.float32)
#     vel = np.zeros(3, dtype=np.float32)
#     return pos, vel


# def circle_traj(t):
#     # Uses your Trajectories.circle_traj
#     return Trajectories.circle_traj(
#         t,
#         radius=0.40,
#         period=20.0,
#         z=1.0
#     )


# # ============================================================
# #   MAIN
# # ============================================================
# def main():
#     USE_CIRCLE = True  # False = pure hover at (0.4, 0, 1)

#     traj_fn = circle_traj if USE_CIRCLE else hover_traj

#     # Let Phoenix own the GUI. We do NOT call pybullet.connect() ourselves.
#     env = DroneFollowTrajectoryEnv(
#         trajectory_fn=traj_fn,
#         control_mode="AttitudeRate",
#         gui=True,                 # Phoenix creates one GUI instance
#         render_mode="human",
#     )
#     env.enable_reset_distribution = False
#     env.domain_randomization = -1

#     ctrl = StableAttRateController()

#     obs, info = env.reset()
#     dt = env.TIME_STEP

#     mode_str = "Circle" if USE_CIRCLE else "Hover"
#     print(f"[INFO] Running: Stable AttitudeRate {mode_str} Flight")

#     try:
#         for step in range(6000):
#             t = env.iteration / env.SIM_FREQ
#             pos_ref, vel_ref = traj_fn(t)
#             state = env.get_state()

#             action = ctrl.act(state, pos_ref, vel_ref, dt)

#             obs, rew, terminated, truncated, info = env.step(action)

#             if step % 200 == 0:
#                 print(
#                     f"t={t:.2f} "
#                     f"pos={state['pos']} "
#                     f"ref={pos_ref} "
#                     f"thrust={action[0]:.3f}"
#                 )

#             if terminated or truncated:
#                 print("[INFO] Episode ended, resetting...")
#                 obs, info = env.reset()

#             time.sleep(dt)

#     finally:
#         # Make sure we cleanly close the client so you don't get the
#         # 'Not connected to physics server' spam next run.
#         if hasattr(env, "bc"):
#             try:
#                 env.bc.disconnect()
#             except Exception:
#                 pass


# if __name__ == "__main__":
#     main()
