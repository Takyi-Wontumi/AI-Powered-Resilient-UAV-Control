import numpy as np

class WebotsCrazyfliePID:
    """
    Full Crazyflie PID port:
      - Vertical velocity loop
      - Altitude PID
      - Velocity XY PID
      - Attitude PID
      - Yaw-rate PID
      - Motor mixer

    Output: motor commands in the range [0, 600]
    """

    def __init__(self):
        # past errors
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0

        # integrators
        self.altitude_integrator = 0.0
        self.vz_integrator = 0.0

        # gains (from Webots)
        self.gains = {
            "kp_att_y":   1.0,
            "kd_att_y":   0.5,
            "kp_att_rp":  0.5,
            "kd_att_rp":  0.1,
            "kp_vel_xy":  2.0,
            "kd_vel_xy":  0.5,

            # Altitude loops
            "kp_z": 10.0,
            "ki_z": 5.0,
            "kd_z": 5.0,

            # Inner fixed-height vertical velocity controller
            "kp_vz": 4.0,
            "ki_vz": 1.0,
            "kd_vz": 1.5
        }

    def pid(self, dt,
            desired_vx, desired_vy, desired_yaw_rate,
            desired_altitude, desired_vz,
            actual_roll, actual_pitch, actual_yaw_rate,
            actual_altitude, actual_vx, actual_vy, actual_vz):
        g = self.gains

        # ========== XY VELOCITY → ATTITUDE ======================
        vx_error = desired_vx - actual_vx
        vy_error = desired_vy - actual_vy

        vx_deriv = (vx_error - self.past_vx_error) / dt
        vy_deriv = (vy_error - self.past_vy_error) / dt

        desired_pitch = g["kp_vel_xy"]*vx_error + g["kd_vel_xy"]*vx_deriv
        desired_roll  = -g["kp_vel_xy"]*vy_error - g["kd_vel_xy"]*vy_deriv

        self.past_vx_error = vx_error
        self.past_vy_error = vy_error

        # ========== FIXED HEIGHT VERTICAL VELOCITY LOOP =========
        vz_error = desired_vz - actual_vz
        vz_deriv = vz_error / dt
        self.vz_integrator += vz_error * dt

        vz_control = (
            g["kp_vz"] * vz_error
            + g["kd_vz"] * vz_deriv
            + g["ki_vz"] * self.vz_integrator
        )

        # ========== ALTITUDE PID (outer loop) ===================
        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.past_alt_error) / dt
        self.altitude_integrator += alt_error * dt

        alt_command = (
            g["kp_z"]*alt_error
            + g["kd_z"]*alt_deriv
            + g["ki_z"]*np.clip(self.altitude_integrator, -2, 2)
            + 48                       # CF baseline hover thrust
            + vz_control               # vertical velocity control
        )

        self.past_alt_error = alt_error

        # ========== ATTITUDE PID ================================
        pitch_error = desired_pitch - actual_pitch
        roll_error  = desired_roll  - actual_roll

        pitch_deriv = (pitch_error - self.past_pitch_error) / dt
        roll_deriv  = (roll_error  - self.past_roll_error)  / dt

        roll_cmd = g["kp_att_rp"]*roll_error + g["kd_att_rp"]*roll_deriv
        pitch_cmd = -g["kp_att_rp"]*pitch_error - g["kd_att_rp"]*pitch_deriv

        self.past_pitch_error = pitch_error
        self.past_roll_error  = roll_error

        # ========== YAW RATE PID ================================
        yaw_rate_error = desired_yaw_rate - actual_yaw_rate
        yaw_cmd = g["kp_att_y"] * yaw_rate_error

        # ========== MOTOR MIXING ===============================
        m1 = alt_command - roll_cmd + pitch_cmd + yaw_cmd
        m2 = alt_command - roll_cmd - pitch_cmd - yaw_cmd
        m3 = alt_command + roll_cmd - pitch_cmd + yaw_cmd
        m4 = alt_command + roll_cmd + pitch_cmd - yaw_cmd

        motors = np.clip([m1, m2, m3, m4], 0, 600)
        return motors
