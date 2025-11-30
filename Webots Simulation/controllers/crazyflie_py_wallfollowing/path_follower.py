from controller import Robot
import numpy as np
from math import atan2, asin


def quat_to_euler(q):
    """Webots gives quaternion as [x,y,z,w]."""
    x, y, z, w = q
    sinr = 2*(w*x + y*z)
    cosr = 1-2*(x*x + y*y)
    roll = atan2(sinr, cosr)

    sinp = 2*(w*y - z*x)
    pitch = np.pi/2 * np.sign(sinp) if abs(sinp)>=1 else asin(sinp)

    siny = 2*(w*z + x*y)
    cosy = 1-2*(y*y + z*z)
    yaw = atan2(siny, cosy)

    return roll, pitch, yaw


class PathFollowerPD:
    def __init__(self):
        self.Kp_xy = np.array([1.0, 1.0])
        self.Kd_xy = np.array([0.8, 0.8])

        self.Kp_z = 2.0
        self.Kd_z = 1.0

        self.Kp_yaw = 2.0
        self.max_angle = 0.35
        self.g = 9.81

    def compute(self, pos, vel, quat, omega, pos_ref, vel_ref):
        roll, pitch, yaw = quat_to_euler(quat)

        # Position errors
        e_pos = pos_ref - pos
        e_vel = vel_ref - vel

        # XY control → acceleration command
        a_des_xy = self.Kp_xy * e_pos[:2] + self.Kd_xy * e_vel[:2]

        # Altitude control
        ez, evz = e_pos[2], e_vel[2]
        a_des_z = self.Kp_z * ez + self.Kd_z * evz

        # Thrust normalization
        thrust = (self.g + a_des_z) / (2*self.g)
        thrust = np.clip(thrust, -1, 1)

        # Horizontal accel → desired roll/pitch
        phi_des  = ( a_des_xy[0]*np.sin(yaw) - a_des_xy[1]*np.cos(yaw) ) / self.g
        theta_des = ( a_des_xy[0]*np.cos(yaw) + a_des_xy[1]*np.sin(yaw) ) / self.g

        phi_des   = np.clip(phi_des,  -self.max_angle, self.max_angle)
        theta_des = np.clip(theta_des, -self.max_angle, self.max_angle)

        yaw_ref = 0.0
        yaw_cmd = self.Kp_yaw*(yaw_ref - yaw)
        yaw_cmd = np.clip(yaw_cmd, -1, 1)

        return thrust, phi_des, theta_des, yaw_cmd
