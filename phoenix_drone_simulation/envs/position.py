import numpy as np
from phoenix_drone_simulation.envs.control import Attitude


class PositionController:
    """
    Implements a simple PD controller for Z-axis altitude control
    and uses the existing Attitude controller as the inner loop.
    Includes a strong anti-climb safety mechanism.
    """

    def __init__(self, drone, bc, time_step):
        self.drone = drone
        self.bc = bc
        self.dt = time_step

        # ===============================
        # PD ALTITUDE GAINS (No Integral Term)
        # ===============================
        # Using the Kp and Kd from the previous PID, but setting Ki to 0.
        self.Kp_z = 16.2  # Proportional gain for altitude error
        self.Kd_z = 11.0  # Derivative gain for velocity error

        # Gravity compensation (essential for altitude control)
        self.g = 9.81

        # Inner attitude controller
        self.att = Attitude(drone=drone, bc=bc, time_step=time_step)

        # Nominal hover thrust in PWM (used as bias)
        self.HOVER_PWM = 18000

        # Max safe PWM to prevent "shooting into space"
        # 55000 was the previous clip max, we'll keep it there but enforce it
        self.MAX_PWM_LIMIT = 55000 
        self.MIN_PWM_LIMIT = 20000

    def reset(self):
        # Only reset the inner attitude controller
        self.att.reset()

    # ================================================================
    #                      MAIN CONTROL ACTION
    # ================================================================
    def act(self, desired_position):
        
        # Current State
        z_des = desired_position[2]
        z = self.drone.xyz[2]
        vz = self.drone.xyz_dot[2]

        # ===============================
        # Z-AXIS PD CONTROL
        # ===============================
        # Position Error
        e_z = z_des - z
        # Velocity Error (target vertical velocity is 0)
        ev_z = -vz
        
        # PD Controller Output (Acceleration Command)
        # u_z = Kp * position_error + Kd * velocity_error
        u_z = (
            self.Kp_z * e_z +
            self.Kd_z * ev_z
        )

        # ===============================
        # THRUST MAPPING 
        # ===============================
        
        # Convert PD output (acceleration command) into PWM thrust.
        # thrust_pwm = HOVER_PWM + Calibration_Factor * u_z
        
        # We'll use the same calibration factor (2000) from the original code
        thrust_pwm = self.HOVER_PWM + 2000 * u_z

        # NOTE: The clamp here is now removed/deferred, as the inner loop 
        # calculation changes individual motor PWMs, overriding this clamp.
        # We normalize based on the *expected* output range of the PD loop.
        
        # Normalize thrust for Attitude controller (required interface)
        # The PD output (thrust_pwm) is converted to a normalized collective thrust command.
        thrust_norm = (thrust_pwm - 30000) / 30000
        thrust_norm = np.clip(thrust_norm, -1.0, 1.0) # Clip normalized value

        # ===============================
        # INNER LOOP EXECUTION (Attitude)
        # ===============================
        # Since we are just hovering, roll, pitch, and yaw targets are zero.
        roll_cmd = 0.0
        pitch_cmd = 0.0
        yaw_cmd = 0.0

        action = np.array([thrust_norm, roll_cmd, pitch_cmd, yaw_cmd])

        # Send command to the cascaded Attitude controller
        # This returns the array of 4 individual motor PWMs.
        pwm = self.att.act(action)

        # ================================================================
        # 🛑 FINAL ANTI-CLIMB SAFETY CLAMP (CRITICAL FIX)
        # ================================================================
        # The clamp must be applied to the final, individual motor PWMs (pwm array)
        # to prevent the attitude correction from pushing the total thrust over the limit.
        
        # Ensure all four motor commands respect the safe limits.
        CLAMP_MAX = 45000 
        CLAMP_MIN = 20000 
        
        pwm = np.clip(pwm, CLAMP_MIN, CLAMP_MAX)

        return pwm