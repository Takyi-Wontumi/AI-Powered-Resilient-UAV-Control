#!/usr/bin/env python3
"""
Extended Kalman Filter (EKF) for drone INS during GPS Dropout
==============================================================
9-state EKF:  [px, py, pz, vx, vy, vz, roll, pitch, yaw]

Nonlinear state transition f(x, u):
  p(k+1)   = p(k) + v(k)*dt
  v(k+1)   = v(k) + [R(att) @ a_body - (0,0,g)] * dt
  att(k+1) = att(k) + omega_body * dt

The Jacobian F = df/dx is computed analytically.  The key nonlinear
coupling is dv/d(att) through the body-to-world rotation matrix R,
which depends on the attitude states (roll, pitch, yaw).

GPS measurement model is linear:  z = H @ x  (position only).

Backward-compatible: state[0:3] = position, state[3:6] = velocity,
so existing code using kf.get_state()[:3] and [:3:6] still works.
"""

import numpy as np


class KalmanFilterINS:
    """
    9-State Extended Kalman Filter for drone INS during GPS dropout.

    State vector:   [px, py, pz, vx, vy, vz, roll, pitch, yaw]
    IMU input:      [ax_body, ay_body, az_body, wx, wy, wz]
    GPS measurement:[px, py, pz]  (position only, linear model)

    The EKF nonlinearity arises from rotating body-frame accelerometer
    readings into the world frame using the current attitude estimate.
    The Jacobian block  F[3:6, 6:9] = dR/d(att) @ a_body * dt  is
    nonzero and attitude-dependent, making this a genuine EKF.
    """

    def __init__(self, dt=0.002):
        self.dt  = dt
        self.g   = 9.81

        # 9-state vector: [px, py, pz, vx, vy, vz, roll, pitch, yaw]
        self.x = np.zeros(9)

        # State covariance (9x9)
        self.P = np.diag([
            0.01, 0.01, 0.01,    # position (m²)
            0.1,  0.1,  0.1,     # velocity (m²/s²)
            0.01, 0.01, 0.01     # attitude (rad²)
        ])

        # Process noise covariance
        self.Q = np.diag([
            1e-5, 1e-5, 1e-5,   # position noise
            5e-3, 5e-3, 5e-3,   # velocity noise (accelerometer bias/noise)
            1e-4, 1e-4, 1e-4    # attitude noise (gyro bias)
        ])

        # GPS measurement noise covariance (m²)
        self.R = np.diag([0.01, 0.01, 0.01])

        # Linear measurement matrix — GPS observes position only
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Latest IMU readings (stored by update_with_imu, used in predict)
        self.accel_body = np.zeros(3)
        self.gyro_body  = np.zeros(3)
        self.dropout_duration_s = 0.0

    # ------------------------------------------------------------------
    # Rotation matrix R (body → world, ZYX convention) and its
    # analytical partial derivatives w.r.t. roll, pitch, yaw.
    # These are the Jacobian blocks that make this a genuine EKF.
    # ------------------------------------------------------------------

    def _rotation_matrix(self, roll, pitch, yaw):
        """Body-to-world rotation matrix (ZYX: R = Rz * Ry * Rx)."""
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,     cp*sr,             cp*cr            ]
        ])

    def _dR_droll(self, roll, pitch, yaw):
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [0,  cy*sp*cr + sy*sr, -cy*sp*sr + sy*cr],
            [0,  sy*sp*cr - cy*sr, -sy*sp*sr - cy*cr],
            [0,  cp*cr,            -cp*sr            ]
        ])

    def _dR_dpitch(self, roll, pitch, yaw):
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [-cy*sp,  cy*cp*sr,  cy*cp*cr],
            [-sy*sp,  sy*cp*sr,  sy*cp*cr],
            [-cp,    -sp*sr,    -sp*cr   ]
        ])

    def _dR_dyaw(self, roll, pitch, yaw):
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        return np.array([
            [-sy*cp, -sy*sp*sr - cy*cr, -sy*sp*cr + cy*sr],
            [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [  0,     0,                 0                ]
        ])

    # ------------------------------------------------------------------
    # EKF steps
    # ------------------------------------------------------------------

    def update_with_imu(self, accel, gyro):
        """Store latest IMU readings.  Called before predict()."""
        self.accel_body = np.asarray(accel, dtype=float)
        self.gyro_body  = np.asarray(gyro,  dtype=float)

    def make_dropout_Q(self, dropout_time_s: float) -> np.ndarray:
        """Increase position/velocity uncertainty as GPS dropout persists."""
        dropout_time_s = max(0.0, float(dropout_time_s))
        Q = self.Q.copy()

        pos_scale = 1.0 + 0.45 * dropout_time_s + 0.35 * dropout_time_s ** 2
        vel_scale = 1.0 + 0.60 * dropout_time_s + 0.15 * dropout_time_s ** 2
        att_scale = 1.0 + 0.20 * dropout_time_s

        Q[0:3, 0:3] *= pos_scale
        Q[3:6, 3:6] *= vel_scale
        Q[6:9, 6:9] *= att_scale
        return Q

    def predict(self, process_noise=None):
        """
        EKF prediction step using nonlinear IMU-driven dynamics.

        Nonlinear state transition f(x):
          p(k+1)   = p(k) + v(k)*dt
          v(k+1)   = v(k) + [R(att) @ a_body - (0,0,g)] * dt
          att(k+1) = att(k) + gyro_body * dt

        Analytical Jacobian F = df/dx (9×9).
        Nonlinear blocks: F[3:6, 6:9] = dR/d(att) @ a_body * dt
        """
        dt               = self.dt
        vx, vy, vz       = self.x[3:6]
        roll, pitch, yaw = self.x[6:9]

        # Body-to-world rotation at current attitude
        R   = self._rotation_matrix(roll, pitch, yaw)
        # World-frame specific force (gravity removed)
        a_w = R @ self.accel_body - np.array([0.0, 0.0, self.g])

        # ----- Nonlinear state propagation -----
        x_new = self.x.copy()
        x_new[0] += vx * dt
        x_new[1] += vy * dt
        x_new[2] += vz * dt
        x_new[3] += a_w[0] * dt
        x_new[4] += a_w[1] * dt
        x_new[5] += a_w[2] * dt
        x_new[6] += self.gyro_body[0] * dt
        x_new[7] += self.gyro_body[1] * dt
        x_new[8] += self.gyro_body[2] * dt
        self.x = x_new

        # ----- Analytical Jacobian F (9×9) -----
        F = np.eye(9)
        # d(pos)/d(vel)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # d(vel)/d(att) — the genuine EKF nonlinear terms
        dR_dr = self._dR_droll(roll, pitch, yaw)
        dR_dp = self._dR_dpitch(roll, pitch, yaw)
        dR_dy = self._dR_dyaw(roll, pitch, yaw)
        F[3:6, 6] = (dR_dr @ self.accel_body) * dt   # d(v)/d(roll)
        F[3:6, 7] = (dR_dp @ self.accel_body) * dt   # d(v)/d(pitch)
        F[3:6, 8] = (dR_dy @ self.accel_body) * dt   # d(v)/d(yaw)

        # Propagate covariance
        Q = self.Q if process_noise is None else np.asarray(process_noise, dtype=float)
        self.P = F @ self.P @ F.T + Q

    def update_with_gps(self, gps_position):
        """
        EKF correction with GPS position measurement.
        Measurement model is linear: z = H @ x (position only).
        """
        z     = np.asarray(gps_position, dtype=float)
        z_hat = self.H @ self.x
        y     = z - z_hat                              # innovation

        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ self.H) @ self.P

    def update_with_baro(self, baro_altitude, variance: float = 0.02 ** 2):
        """Scalar altitude correction from a barometer or range sensor."""
        z = np.array([float(baro_altitude)], dtype=float)
        H = np.zeros((1, 9), dtype=float)
        H[0, 2] = 1.0
        R = np.array([[float(variance)]], dtype=float)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def update_with_velocity(self, velocity, variance_xy: float = 0.04 ** 2, variance_z: float = 0.06 ** 2):
        """Velocity correction for odometry / optical-flow-derived motion."""
        z = np.asarray(velocity, dtype=float).reshape(3)
        H = np.zeros((3, 9), dtype=float)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0
        R = np.diag([float(variance_xy), float(variance_xy), float(variance_z)])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_position(self):
        """Position estimate [px, py, pz]."""
        return self.x[0:3].copy()

    def get_velocity(self):
        """Velocity estimate [vx, vy, vz]."""
        return self.x[3:6].copy()

    def get_attitude(self):
        """Attitude estimate [roll, pitch, yaw] in radians."""
        return self.x[6:9].copy()

    def get_state(self):
        """Full 9-state vector [px,py,pz, vx,vy,vz, roll,pitch,yaw]."""
        return self.x.copy()

    def set_state(self, pos, vel, rpy=None):
        """Initialise or hard-resync the filter state."""
        self.x[0:3] = np.asarray(pos)
        self.x[3:6] = np.asarray(vel)
        if rpy is not None:
            self.x[6:9] = np.asarray(rpy)
        self.dropout_duration_s = 0.0
        self.P = np.diag([
            1e-3, 1e-3, 1e-3,
            1e-2, 1e-2, 1e-2,
            1e-3, 1e-3, 1e-3
        ])

    def reset_to_gps(self, gps_position, velocity, rpy=None):
        """Hard-reset position/velocity at the start of a GPS dropout."""
        self.set_state(gps_position, velocity, rpy)
