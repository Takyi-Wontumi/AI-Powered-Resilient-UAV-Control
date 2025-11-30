import numpy as np

class PDPositionController:
    """
    Outer-loop PD controller for Crazyflie in ATTITUDE mode.
    Matches the structure of your QuadcopterPD control law.
    """

    def __init__(self):
        # Position gains (outer loop)
        self.Kp_xy = np.array([0.8, 0.8])
        self.Kd_xy = np.array([0.6, 0.6])

        # Altitude gains
        self.Kp_z = 3.0
        self.Kd_z = 1.2

        # Gravity
        self.g = 9.81

        # Hover thrust (normalized for attitude mode)
        self.hover_thrust = 0.48
        self.thrust_scale = 0.08  # small, non-aggressive

    def compute_action(self, state, pos_ref, vel_ref):
        """
        state: a dict from DroneFollowPathEnv
            state["pos"], state["vel"], state["rpy"], ...
        Returns the ATTITUDE mode action:
            [phi_des, theta_des, yaw_rate_des, thrust_norm]
        """

        pos = state["pos"]          # drone.xyz
        vel = state["vel"]          # drone.xyz_dot
        rpy = state["rpy"]          # roll, pitch, yaw
        yaw = rpy[2]

        # Position & velocity errors
        exy = pos_ref[:2] - pos[:2]
        evxy = vel_ref[:2] - vel[:2]

        ez = pos_ref[2] - pos[2]
        evz = vel_ref[2] - vel[2]

        # Desired horizontal acceleration
        a_des_xy = self.Kp_xy * exy + self.Kd_xy * evxy

        # Desired vertical acceleration + gravity
        a_des_z = self.Kp_z * ez + self.Kd_z * evz + self.g

        # Convert desired acceleration → desired roll/pitch
        phi_des = ( a_des_xy[0] * np.sin(yaw) - a_des_xy[1] * np.cos(yaw) ) / self.g
        theta_des = ( a_des_xy[0] * np.cos(yaw) + a_des_xy[1] * np.sin(yaw) ) / self.g

        # Normalize thrust for Phoenix attitude mode
        thrust_norm = self.hover_thrust + self.thrust_scale * (a_des_z - self.g)

        # Clip to safe range
        thrust_norm = np.clip(thrust_norm, 0.25, 0.75)

        # No yaw control yet → yaw rate 0
        yaw_rate_des = 0.0

        return np.array([phi_des, theta_des, yaw_rate_des, thrust_norm])
