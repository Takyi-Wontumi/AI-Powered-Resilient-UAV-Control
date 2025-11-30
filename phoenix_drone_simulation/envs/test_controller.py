import numpy as np
from phoenix_drone_simulation.envs.control import Control
from examples.pid_webots_full import WebotsCrazyfliePID

class WebotsFullPID(Control):
    def __init__(self, drone, bc, time_step):
        super().__init__(drone, bc, time_step)
        self.pid = WebotsCrazyfliePID()

    def reset(self):
        super().reset()
        self.pid = WebotsCrazyfliePID()

    def act(self, action, **kwargs):

        # ============================================
        # CASE 1: Phoenix internal call (wrong action)
        # Phoenix will ALWAYS call act() with a (4,) vector.
        # We MUST bypass this, otherwise PID crashes.
        # ============================================
        if len(action) == 4:
            return action  # already motor PWMs → pass through

        # ============================================
        # CASE 2: Our command from follow_path_test3.py
        # action = [vx, vy, yaw_rate, altitude, vz]
        # ============================================
        desired_vx, desired_vy, desired_yaw, desired_alt, desired_vz = action

        dt = self.time_step

        roll, pitch, yaw = self.drone.rpy
        yaw_rate = self.drone.rpy_dot[2]
        vx, vy, vz = self.drone.xyz_dot
        z = self.drone.xyz[2]

        motors = self.pid.pid(
            dt,
            desired_vx, desired_vy, desired_yaw,
            desired_alt, desired_vz,
            actual_roll=roll,
            actual_pitch=pitch,
            actual_yaw_rate=yaw_rate,
            actual_altitude=z,
            actual_vx=vx,
            actual_vy=vy,
            actual_vz=vz
        )

        # Scale [0,600] → Phoenix [0,60000]
        PWMs = (np.array(motors) / 600.0 * 60000.0).astype(np.float32)
        return PWMs
