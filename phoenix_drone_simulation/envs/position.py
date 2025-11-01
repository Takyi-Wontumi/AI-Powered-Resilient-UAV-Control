import numpy as np
from phoenix_drone_simulation.envs.control import Attitude

class PositionController:
    """High-level PID position controller with attitude inner loop."""

    def __init__(self, drone, bc, time_step):
        self.drone = drone
        self.bc = bc
        self.time_step = time_step
        self.attitude = Attitude(drone=drone, bc=bc, time_step=time_step)

        # Tuned PID gains for a stable Crazyflie hover
        self.kp = np.array([2.5, 2.5, 4.0])
        self.ki = np.array([0.01, 0.01, 0.02])
        self.kd = np.array([1.0, 1.0, 2.0])

        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.gravity = 9.81

    def reset(self):
        self.integral[:] = 0
        self.prev_error[:] = 0
        self.attitude.reset()

    def act(self, desired_position):
        """Computes thrust and attitude from desired 3D position."""
        # === Current state ===
        pos = np.array(self.drone.xyz)
        vel = np.array(self.drone.xyz_dot)

        # === Errors ===
        e_p = desired_position - pos
        e_v = -vel
        self.integral += e_p * self.time_step

        # === PID acceleration command ===
        acc_cmd = (
            self.kp * e_p
            + self.kd * e_v
            + self.ki * self.integral
        )

        # Gravity compensation
        acc_cmd[2] += self.gravity

        # === Convert acceleration to attitude & thrust ===
        roll_des  = np.clip(acc_cmd[1] / self.gravity, -0.4, 0.4)
        pitch_des = np.clip(-acc_cmd[0] / self.gravity, -0.4, 0.4)

        # Scale thrust roughly into [0,1] range (empirical for Crazyflie)
        thrust = np.clip(acc_cmd[2] / (2 * self.gravity), 0.0, 1.0)

        # === Send to inner attitude controller ===
        action = np.array([thrust, roll_des, pitch_des, 0.0])
        pwm = self.attitude.act(action)
        return pwm
