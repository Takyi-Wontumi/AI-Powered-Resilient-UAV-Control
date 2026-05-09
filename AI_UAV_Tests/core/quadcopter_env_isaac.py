import numpy as np


class IsaacFriendlyQuadcopterPID:
    """
    Isaac-oriented quadcopter controller for direct per-motor force output.

    This controller is intentionally simpler than the Phoenix/Crazyflie-style
    cascaded controller in `quadcopter_env.py`:
    - no inner body-rate PID loop
    - direct motor-force mixing around hover thrust
    - yaw hold disabled by default to reduce XY drift from yaw-coupling

    Assumptions:
    - `ang` is [roll, pitch, yaw] in radians
    - `rate` is body-frame [p, q, r] in rad/s
    - returned motor order is:
      [front_left, front_right, rear_right, rear_left]
    """

    def __init__(
        self,
        dt=0.002,
        g=9.81,
        mass=0.028,
        max_angle_deg=20.0,
        max_motor_force=0.20,
        yaw_hold=False,
    ):
        self.dt = float(dt)
        self.g = float(g)
        self.m = float(mass)
        self.max_angle = np.deg2rad(float(max_angle_deg))
        self.max_motor_force = float(max_motor_force)
        self.yaw_hold = bool(yaw_hold)

        # Position loop gains. These are intentionally closer to the
        # Isaac graph version the user reported as stable.
        self.Kp_xy = np.array([8.0, 8.0], dtype=float)
        self.Kd_xy = np.array([10.0, 10.0], dtype=float)

        self.Kp_z = 14.2
        self.Ki_z = 2.50175
        self.Kd_z = 12.3494
        self.z_int_limit = 1.0

        # Attitude loop gains driving direct mixer offsets.
        self.Kp_att = np.array([8.0, 8.0, 4.0], dtype=float)
        self.Ki_att = np.array([1.0, 1.0, 0.5], dtype=float)
        self.att_int_limit = np.array([0.3, 0.3, 0.3], dtype=float)

        # Scale attitude error into per-motor force offsets. These are not
        # physical torques; they are small force corrections for the mixer.
        self.att_force_scale = np.array([1.0e-4, 1.0e-4, 6.0e-5], dtype=float)

        # Light angular-rate damping to avoid oscillation in a direct-force loop.
        self.rate_damping = np.array([5.0e-5, 5.0e-5, 2.5e-5], dtype=float)

        self.reset()

    def reset(self):
        self.x = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.ang = np.zeros(3, dtype=float)
        self.rate = np.zeros(3, dtype=float)

        self.z_int = 0.0
        self.att_int = np.zeros(3, dtype=float)

    def inject_external_state(self, x, v, ang, rate):
        self.x = np.asarray(x, dtype=float).copy()
        self.v = np.asarray(v, dtype=float).copy()
        self.ang = np.asarray(ang, dtype=float).copy()
        self.rate = np.asarray(rate, dtype=float).copy()

    @staticmethod
    def wrap_angle(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def position_pid(self, pos_ref, vel_ref, z_ref=None, yaw_ref=None):
        pos_ref = np.asarray(pos_ref, dtype=float)
        vel_ref = np.asarray(vel_ref, dtype=float)
        if z_ref is None:
            z_ref = float(pos_ref[2])

        exy = pos_ref[:2] - self.x[:2]
        evxy = vel_ref[:2] - self.v[:2]
        a_xy = self.Kp_xy * exy + self.Kd_xy * evxy
        a_xy = np.clip(a_xy, -3.0, 3.0)

        ez = float(z_ref) - self.x[2]
        evz = float(vel_ref[2]) - self.v[2]
        self.z_int = np.clip(self.z_int + ez * self.dt, -self.z_int_limit, self.z_int_limit)

        a_z = self.Kp_z * ez + self.Kd_z * evz + self.Ki_z * self.z_int
        a_z = np.clip(a_z, -4.0, 4.0)

        thrust = self.m * (self.g + a_z)
        thrust = np.clip(thrust, 0.5 * self.m * self.g, 1.5 * self.m * self.g)

        psi = self.ang[2]
        ax, ay = a_xy
        roll_des = (ax * np.sin(psi) - ay * np.cos(psi)) / self.g
        pitch_des = (ax * np.cos(psi) + ay * np.sin(psi)) / self.g

        roll_des = np.clip(roll_des, -self.max_angle, self.max_angle)
        pitch_des = np.clip(pitch_des, -self.max_angle, self.max_angle)

        if yaw_ref is None:
            yaw_des = self.ang[2] if not self.yaw_hold else 0.0
        else:
            yaw_des = float(yaw_ref)

        return thrust, roll_des, pitch_des, yaw_des

    def attitude_force_offsets(self, roll_des, pitch_des, yaw_des):
        yaw_error = self.wrap_angle(yaw_des - self.ang[2])
        e_att = np.array(
            [
                float(roll_des) - self.ang[0],
                float(pitch_des) - self.ang[1],
                yaw_error,
            ],
            dtype=float,
        )

        if not self.yaw_hold and np.isclose(yaw_des, self.ang[2]):
            e_att[2] = 0.0

        self.att_int += e_att * self.dt
        self.att_int = np.clip(self.att_int, -self.att_int_limit, self.att_int_limit)

        offsets = (self.Kp_att * e_att + self.Ki_att * self.att_int) * self.att_force_scale
        offsets -= self.rate_damping * self.rate

        if not self.yaw_hold and np.isclose(yaw_des, self.ang[2]):
            offsets[2] = 0.0

        return offsets, e_att

    def mixer(self, thrust, offsets):
        roll_term, pitch_term, yaw_term = np.asarray(offsets, dtype=float)
        f_base = float(thrust) / 4.0

        # X configuration. Remap externally if your Isaac asset uses a different
        # rotor order than [FL, FR, RR, RL].
        f1 = f_base - roll_term + pitch_term + yaw_term
        f2 = f_base + roll_term + pitch_term - yaw_term
        f3 = f_base + roll_term - pitch_term + yaw_term
        f4 = f_base - roll_term - pitch_term - yaw_term

        motor_forces = np.array([f1, f2, f3, f4], dtype=float)
        return np.clip(motor_forces, 0.0, self.max_motor_force)

    def force_vectors(self, motor_forces):
        motor_forces = np.asarray(motor_forces, dtype=float)
        return [(0.0, 0.0, float(force)) for force in motor_forces]

    def step(self, pos_ref, vel_ref, z_ref=None, yaw_ref=None):
        thrust, roll_des, pitch_des, yaw_des = self.position_pid(
            pos_ref=pos_ref,
            vel_ref=vel_ref,
            z_ref=z_ref,
            yaw_ref=yaw_ref,
        )
        offsets, e_att = self.attitude_force_offsets(
            roll_des=roll_des,
            pitch_des=pitch_des,
            yaw_des=yaw_des,
        )
        motor_forces = self.mixer(thrust=thrust, offsets=offsets)

        return {
            "thrust_cmd": float(thrust),
            "motor_forces": motor_forces,
            "force_vectors": self.force_vectors(motor_forces),
            "offset_cmd": offsets,
            "att_error": e_att,
            "ang_des": np.array([roll_des, pitch_des, yaw_des], dtype=float),
            "x": self.x.copy(),
            "v": self.v.copy(),
            "ang": self.ang.copy(),
            "rate": self.rate.copy(),
        }


def example_usage():
    controller = IsaacFriendlyQuadcopterPID()
    controller.inject_external_state(
        x=np.zeros(3),
        v=np.zeros(3),
        ang=np.zeros(3),
        rate=np.zeros(3),
    )
    ctrl = controller.step(
        pos_ref=np.array([0.0, 0.0, 1.0]),
        vel_ref=np.zeros(3),
        z_ref=1.0,
    )
    return ctrl


if __name__ == "__main__":
    ctrl = example_usage()
    print("Isaac-friendly controller example")
    print(f"thrust_cmd = {ctrl['thrust_cmd']:.5f} N")
    print(
        "motor_forces = "
        f"{ctrl['motor_forces'][0]:.5f}, "
        f"{ctrl['motor_forces'][1]:.5f}, "
        f"{ctrl['motor_forces'][2]:.5f}, "
        f"{ctrl['motor_forces'][3]:.5f} N"
    )
