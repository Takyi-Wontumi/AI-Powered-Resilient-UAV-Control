"""Extended Kalman filter for the quadcopter state used in AI_UAV_Tests.

State ordering matches the controller/env files:
    [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]

The filter is intentionally standalone:
    - `predict(omega)` propagates the nonlinear quadcopter dynamics
    - `update(z, H)` performs a generic linear measurement update
    - helper methods build common measurement models for subsets of the state

This keeps the EKF usable with PyBullet, Isaac, Phoenix, or the internal RK4
sim without forcing one sensor stack.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from AI_UAV_Tests.sensors_ekf import EKFSensorNoise


STATE_DIM = 12

POS_IDX = slice(0, 3)
VEL_IDX = slice(3, 6)
ANG_IDX = slice(6, 9)
RATE_IDX = slice(9, 12)

_EKF_Q_SCALE = 1.0
_EKF_R_SCALE = 1.0

def _uniform_variance(half_range: float) -> float:
    """Variance of U[-a, a] when `half_range == a`."""
    a = float(half_range)
    return (a * a) / 3.0


def _sensor_measurement_noise_diag(sensor_noise: EKFSensorNoise, dt: float) -> np.ndarray:
    """Map the simulator sensor model onto the EKF's direct-state measurement R.

    Position, velocity, and attitude are direct white-noise mappings.
    For body rates, match only the terms that appear per sample in the selected
    EKF sensor model. When `sample_turn_on_bias_once=True`, turn-on bias is a
    constant offset per reset and is intentionally excluded from white R.
    """
    pos_var = sensor_noise.pos_norm_std ** 2 + _uniform_variance(sensor_noise.pos_unif_range)
    vel_var = sensor_noise.vel_norm_std ** 2 + _uniform_variance(sensor_noise.vel_unif_range)
    ang_var = sensor_noise.quat_norm_std ** 2 + _uniform_variance(sensor_noise.quat_unif_range)

    sigma_g_d = sensor_noise.gyro_noise_density / np.sqrt(float(dt))
    rate_var = (
        sigma_g_d ** 2
        + sensor_noise.gyro_random_walk ** 2
    )
    if not getattr(sensor_noise, "sample_turn_on_bias_once", False):
        rate_var += sensor_noise.gyro_turn_on_bias_sigma ** 2

    return np.array(
        [pos_var, pos_var, pos_var,
         vel_var, vel_var, vel_var,
         ang_var, ang_var, ang_var,
         rate_var, rate_var, rate_var],
        dtype=float,
    )


def _continuous_white_accel_block(dt: float, sigma: float) -> np.ndarray:
    """Continuous white acceleration noise integrated into one [state, rate] pair."""
    q = float(sigma) ** 2
    return q * np.array(
        [
            [dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 2 / 2.0, dt],
        ],
        dtype=float,
    )


def _build_structured_process_noise(
    dt: float,
    sigma_acc_xy: float,
    sigma_acc_z: float,
    sigma_ang_rate_xy: float,
    sigma_ang_rate_z: float,
    q_pos_floor: float = 5.0e-6,
    q_vel_floor: float = 5.0e-4,
    q_ang_floor: float = 1.0e-4,
    q_rate_floor: float = 2.0e-3,
) -> np.ndarray:
    """Build a structured Q for translational and angular integrated states."""
    Q = np.zeros((STATE_DIM, STATE_DIM), dtype=float)

    lin_xy = _continuous_white_accel_block(dt, sigma_acc_xy)
    lin_z = _continuous_white_accel_block(dt, sigma_acc_z)
    ang_xy = _continuous_white_accel_block(dt, sigma_ang_rate_xy)
    ang_z = _continuous_white_accel_block(dt, sigma_ang_rate_z)

    for pos_idx, vel_idx, block in ((0, 3, lin_xy), (1, 4, lin_xy), (2, 5, lin_z)):
        Q[np.ix_([pos_idx, vel_idx], [pos_idx, vel_idx])] = block
        Q[pos_idx, pos_idx] += float(q_pos_floor)
        Q[vel_idx, vel_idx] += float(q_vel_floor)

    for ang_idx, rate_idx, block in ((6, 9, ang_xy), (7, 10, ang_xy), (8, 11, ang_z)):
        Q[np.ix_([ang_idx, rate_idx], [ang_idx, rate_idx])] = block
        Q[ang_idx, ang_idx] += float(q_ang_floor)
        Q[rate_idx, rate_idx] += float(q_rate_floor)

    return Q * _EKF_Q_SCALE


@dataclass(frozen=True)
class QuadcopterPhysicalParams:
    """Defaults aligned with the Phoenix Bullet Crazyflie URDF."""

    m: float = 0.030
    # Phoenix uses an X-configuration; 0.028 m is the effective torque arm.
    l: float = 0.028
    b: float = 1.4e-6
    d: float = 1.1e-7
    g: float = 9.81
    Ix: float = 1.33e-5
    Iy: float = 1.33e-5
    Iz: float = 2.64e-5
    # Leave drag disabled in the EKF until a process model consistent with the
    # Phoenix drag implementation is identified.
    drag_x: float = 0.0
    drag_y: float = 0.0
    drag_z: float = 0.0


class QuadcopterEKF:
    """12-state EKF aligned with the quadcopter env/controller state."""

    def __init__(
        self,
        dt: float = 0.002,
        params: QuadcopterPhysicalParams | None = None,
        process_noise_diag: Sequence[float] | None = None,
        measurement_noise_diag: Sequence[float] | None = None,
        initial_cov_diag: Sequence[float] | None = None,
        adapt_noise: bool = False,
        sigma_acc_xy: float = 0.65,
        sigma_acc_z: float = 1.0,
        sigma_ang_rate_xy: float = 0.4,
        sigma_ang_rate_z: float = 0.6,
        q_pos_floor: float = 5.0e-6,
        q_vel_floor: float = 5.0e-4,
        q_ang_floor: float = 2.0e-4,
        q_rate_floor: float = 3.0e-3,
    ):
        self.dt = float(dt)
        self.params = params or QuadcopterPhysicalParams()
        self.adapt_noise = adapt_noise
        self.sigma_acc_xy = float(sigma_acc_xy)
        self.sigma_acc_z = float(sigma_acc_z)
        self.sigma_ang_rate_xy = float(sigma_ang_rate_xy)
        self.sigma_ang_rate_z = float(sigma_ang_rate_z)
        self.q_pos_floor = float(q_pos_floor)
        self.q_vel_floor = float(q_vel_floor)
        self.q_ang_floor = float(q_ang_floor)
        self.q_rate_floor = float(q_rate_floor)
        self._uses_custom_process_noise = process_noise_diag is not None

        if process_noise_diag is None:
            process_noise = self._default_process_noise(self.dt)
        else:
            process_noise = np.asarray(process_noise_diag, dtype=float)
            if process_noise.shape == (STATE_DIM,):
                process_noise = np.diag(process_noise)
            elif process_noise.shape != (STATE_DIM, STATE_DIM):
                raise ValueError("process_noise_diag must have shape (12,) or (12, 12)")

        if measurement_noise_diag is None:
            measurement_noise_diag = _sensor_measurement_noise_diag(
                EKFSensorNoise(bypass=True), self.dt
            ) * _EKF_R_SCALE

        self.R_default = np.diag(np.asarray(measurement_noise_diag, dtype=float))
        self.Q = process_noise.copy()

        if initial_cov_diag is None:
            initial_cov_diag = np.array(
                [
                    0.05,
                    0.05,
                    0.05,
                    0.10,
                    0.10,
                    0.10,
                    0.05,
                    0.05,
                    0.05,
                    0.10,
                    0.10,
                    0.10,
                ],
                dtype=float,
            )

        self.P0 = np.diag(np.asarray(initial_cov_diag, dtype=float))
        self.x = np.zeros(STATE_DIM, dtype=float)
        self.P = self.P0.copy()
        self.last_innovation: np.ndarray | None = None
        self.last_S: np.ndarray | None = None
        self.last_NIS: float | None = None
        self.last_meas_dim: int | None = None
        self.last_mahalanobis: float | None = None
        self.last_update_accepted: bool | None = None
        self.last_gate_threshold: float | None = None

    def _default_process_noise(self, dt: float) -> np.ndarray:
        return _build_structured_process_noise(
            dt=float(dt),
            sigma_acc_xy=self.sigma_acc_xy,
            sigma_acc_z=self.sigma_acc_z,
            sigma_ang_rate_xy=self.sigma_ang_rate_xy,
            sigma_ang_rate_z=self.sigma_ang_rate_z,
            q_pos_floor=self.q_pos_floor,
            q_vel_floor=self.q_vel_floor,
            q_ang_floor=self.q_ang_floor,
            q_rate_floor=self.q_rate_floor,
        )

    def reset(
        self,
        state: Sequence[float] | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        if state is None:
            self.x = np.zeros(STATE_DIM, dtype=float)
        else:
            self.x = np.asarray(state, dtype=float).reshape(STATE_DIM).copy()

        if covariance is None:
            self.P = self.P0.copy()
        else:
            cov = np.asarray(covariance, dtype=float)
            if cov.shape != (STATE_DIM, STATE_DIM):
                raise ValueError("covariance must have shape (12, 12)")
            self.P = cov.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[POS_IDX].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[VEL_IDX].copy()

    @property
    def attitude(self) -> np.ndarray:
        return self.x[ANG_IDX].copy()

    @property
    def body_rates(self) -> np.ndarray:
        return self.x[RATE_IDX].copy()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _wrap_state_angles(self, state: np.ndarray) -> np.ndarray:
        wrapped = np.asarray(state, dtype=float).copy()
        wrapped[ANG_IDX] = [self._wrap_angle(angle) for angle in wrapped[ANG_IDX]]
        return wrapped

    @staticmethod
    def _rx(phi: float) -> np.ndarray:
        c = np.cos(phi)
        s = np.sin(phi)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

    @staticmethod
    def _rz(psi: float) -> np.ndarray:
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def rotation_matrix(self, angles: Sequence[float]) -> np.ndarray:
        phi, theta, psi = np.asarray(angles, dtype=float)
        return self._rz(psi) @ self._ry(theta) @ self._rx(phi)

    def dynamics(self, state: Sequence[float], omega: Sequence[float]) -> np.ndarray:
        """Continuous-time quadcopter dynamics for the EKF process model."""
        params = self.params
        state = np.asarray(state, dtype=float)
        omega = np.asarray(omega, dtype=float).reshape(4)

        vx, vy, vz = state[VEL_IDX]
        phi, theta, _psi = state[ANG_IDX]
        p, q, r = state[RATE_IDX]

        rotation = self.rotation_matrix(state[ANG_IDX])
        thrust = params.b * np.sum(omega ** 2)
        drag = np.array(
            [
                params.drag_x * vx,
                params.drag_y * vy,
                params.drag_z * vz,
            ],
            dtype=float,
        )
        acc = (
            thrust * (rotation @ np.array([0.0, 0.0, 1.0], dtype=float)) / params.m
            - np.array([0.0, 0.0, params.g], dtype=float)
            - drag
        )

        w1, w2, w3, w4 = omega
        tau_phi = params.l * params.b * (w4 ** 2 - w2 ** 2)
        tau_theta = params.l * params.b * (w3 ** 2 - w1 ** 2)
        tau_psi = params.d * (w1 ** 2 - w2 ** 2 + w3 ** 2 - w4 ** 2)

        p_dot = ((params.Iy - params.Iz) / params.Ix) * q * r + tau_phi / params.Ix
        q_dot = ((params.Iz - params.Ix) / params.Iy) * p * r + tau_theta / params.Iy
        r_dot = ((params.Ix - params.Iy) / params.Iz) * p * q + tau_psi / params.Iz

        ctheta = np.clip(np.cos(theta), 1.0e-3, None)
        stheta = np.sin(theta)
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        euler_rate_map = np.array(
            [
                [1.0, sphi * stheta / ctheta, cphi * stheta / ctheta],
                [0.0, cphi, -sphi],
                [0.0, sphi / ctheta, cphi / ctheta],
            ],
            dtype=float,
        )
        ang_dot = euler_rate_map @ np.array([p, q, r], dtype=float)

        return np.array(
            [
                vx,
                vy,
                vz,
                acc[0],
                acc[1],
                acc[2],
                ang_dot[0],
                ang_dot[1],
                ang_dot[2],
                p_dot,
                q_dot,
                r_dot,
            ],
            dtype=float,
        )

    def _rk4_step(self, state: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        f = lambda s: self.dynamics(s, omega)
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        predicted = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._wrap_state_angles(predicted)

    def _numerical_jacobian(self, func, x0: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float)
        y0 = np.asarray(func(x0), dtype=float)
        jac = np.zeros((y0.size, x0.size), dtype=float)

        for idx in range(x0.size):
            delta = np.zeros_like(x0)
            delta[idx] = eps
            y_plus = np.asarray(func(x0 + delta), dtype=float)
            y_minus = np.asarray(func(x0 - delta), dtype=float)
            diff = y_plus - y_minus
            for angle_idx in range(ANG_IDX.start, min(ANG_IDX.stop, diff.size)):
                diff[angle_idx] = self._wrap_angle(diff[angle_idx])
            jac[:, idx] = diff / (2.0 * eps)

        return jac

    def _stabilize_covariance(
        self,
        min_var: float = 1.0e-9,
        max_var: float = 1.0e3,
    ) -> None:
        self.P = 0.5 * (self.P + self.P.T)
        diag = np.clip(np.diag(self.P), float(min_var), float(max_var))
        np.fill_diagonal(self.P, diag)

        eigvals = np.linalg.eigvalsh(self.P)
        min_eig = float(np.min(eigvals))
        if min_eig < min_var:
            self.P += np.eye(self.P.shape[0], dtype=float) * (float(min_var) - min_eig + 1.0e-12)

    def predict(
        self,
        omega: Sequence[float],
        dt: float | None = None,
        process_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the EKF process update using motor angular speeds."""
        dt = self.dt if dt is None else float(dt)
        omega = np.asarray(omega, dtype=float).reshape(4)
        if process_noise is None:
            process_noise = self.Q if self._uses_custom_process_noise else self._default_process_noise(dt)
        else:
            process_noise = np.asarray(process_noise, dtype=float)
        if process_noise.shape != (STATE_DIM, STATE_DIM):
            raise ValueError("process_noise must have shape (12, 12)")

        transition = lambda state: self._rk4_step(state, omega, dt)
        F = self._numerical_jacobian(transition, self.x)

        self.x = transition(self.x)
        self.P = F @ self.P @ F.T + process_noise
        self._stabilize_covariance()
        return self.x.copy()

    def make_dropout_Q(self, dt: float | None = None, dropout_time: float = 0.0) -> np.ndarray:
        """Inflate process noise by state group during exteroceptive dropout."""
        dt = self.dt if dt is None else float(dt)
        dropout_time = max(0.0, float(dropout_time))
        Q = self.Q.copy() if self._uses_custom_process_noise else self._default_process_noise(dt)

        xy_pos_scale = 1.0 + 5.0 * dropout_time + 8.0 * dropout_time ** 2
        z_pos_scale = 1.0 + 0.8 * dropout_time + 0.6 * dropout_time ** 2
        xy_vel_scale = 1.0 + 6.0 * dropout_time + 5.0 * dropout_time ** 2
        z_vel_scale = 1.0 + 1.0 * dropout_time + 0.4 * dropout_time ** 2
        att_scale = 1.0 + 1.10 * dropout_time
        rate_scale = 1.0 + 1.40 * dropout_time
        yaw_scale = 1.0 + 1.75 * dropout_time

        Q[0:2, 0:2] *= xy_pos_scale
        Q[2, 2] *= z_pos_scale
        Q[3:5, 3:5] *= xy_vel_scale
        Q[5, 5] *= z_vel_scale
        Q[6:9, 6:9] *= att_scale
        Q[9:12, 9:12] *= rate_scale
        Q[8, 8] *= yaw_scale
        Q[11, 11] *= yaw_scale
        return Q

    def predict_dropout(
        self,
        omega: Sequence[float],
        dt: float | None = None,
        dropout_time: float = 0.0,
    ) -> np.ndarray:
        """Prediction helper for dropout periods using state-group-specific Q inflation."""
        Q_dropout = self.make_dropout_Q(dt=dt, dropout_time=dropout_time)
        return self.predict(omega=omega, dt=dt, process_noise=Q_dropout)

    @staticmethod
    def measurement_matrix(indices: Iterable[int]) -> np.ndarray:
        indices = tuple(int(idx) for idx in indices)
        H = np.zeros((len(indices), STATE_DIM), dtype=float)
        for row, idx in enumerate(indices):
            if idx < 0 or idx >= STATE_DIM:
                raise ValueError("measurement index out of range")
            H[row, idx] = 1.0
        return H

    def default_measurement_noise(self, indices: Iterable[int]) -> np.ndarray:
        indices = tuple(int(idx) for idx in indices)
        return np.diag(np.diag(self.R_default)[list(indices)])

    @staticmethod
    def stack_measurements(
        *components: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate sensor-specific measurement blocks into one EKF update."""
        valid_components = [component for component in components if component is not None]
        if not valid_components:
            raise ValueError("at least one measurement component must be provided")

        z = np.concatenate([np.asarray(component[0], dtype=float).reshape(-1) for component in valid_components])
        H = np.vstack([np.asarray(component[1], dtype=float) for component in valid_components])
        total_dim = int(sum(np.asarray(component[0], dtype=float).size for component in valid_components))
        R = np.zeros((total_dim, total_dim), dtype=float)
        row = 0
        for component in valid_components:
            block = np.asarray(component[2], dtype=float)
            next_row = row + block.shape[0]
            R[row:next_row, row:next_row] = block
            row = next_row
        return z, H, R

    def build_gps_measurement(self, position: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_measurement(position=position)

    def build_velocity_measurement(self, velocity: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_measurement(velocity=velocity)

    def build_odom_measurement(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        *,
        position_std: float | Sequence[float] | None = None,
        velocity_std: float | Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measurement = np.concatenate(
            [
                np.asarray(position, dtype=float).reshape(3),
                np.asarray(velocity, dtype=float).reshape(3),
            ]
        )
        H = self.measurement_matrix([0, 1, 2, 3, 4, 5])
        R = self.default_measurement_noise([0, 1, 2, 3, 4, 5])

        if position_std is not None:
            pos_std = np.asarray(position_std, dtype=float)
            if pos_std.ndim == 0:
                pos_std = np.full(3, float(pos_std), dtype=float)
            R[0:3, 0:3] = np.diag(np.square(pos_std.reshape(3)))
        if velocity_std is not None:
            vel_std = np.asarray(velocity_std, dtype=float)
            if vel_std.ndim == 0:
                vel_std = np.full(3, float(vel_std), dtype=float)
            R[3:6, 3:6] = np.diag(np.square(vel_std.reshape(3)))
        return measurement, H, R

    def build_baro_measurement(
        self,
        z: float,
        std: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measurement = np.array([float(z)], dtype=float)
        H = self.measurement_matrix([2])
        if std is None:
            R = self.default_measurement_noise([2])
        else:
            R = np.array([[float(std) ** 2]], dtype=float)
        return measurement, H, R

    def build_gyro_measurement(self, rates: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_measurement(rates=rates)

    def build_attitude_measurement(self, attitude: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.build_measurement(attitude=attitude)

    def build_velocity_pseudo_measurement(
        self,
        velocity_ref: Sequence[float],
        std_xy: float = 0.10,
        std_z: float = 0.08,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measurement = np.asarray(velocity_ref, dtype=float).reshape(3)
        H = self.measurement_matrix([3, 4, 5])
        R = np.diag([float(std_xy) ** 2, float(std_xy) ** 2, float(std_z) ** 2])
        return measurement, H, R

    def build_position_pseudo_measurement(
        self,
        position_ref: Sequence[float],
        std_xy: float = 0.08,
        std_z: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measurement = np.asarray(position_ref, dtype=float).reshape(3)
        H = self.measurement_matrix([0, 1, 2])
        R = np.diag([float(std_xy) ** 2, float(std_xy) ** 2, float(std_z) ** 2])
        return measurement, H, R

    def build_optical_flow_measurement(
        self,
        velocity_xy: Sequence[float],
        std_xy: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        measurement = np.asarray(velocity_xy, dtype=float).reshape(2)
        H = self.measurement_matrix([3, 4])
        R = np.diag([float(std_xy) ** 2, float(std_xy) ** 2])
        return measurement, H, R

    def build_measurement(
        self,
        *,
        position: Sequence[float] | None = None,
        velocity: Sequence[float] | None = None,
        attitude: Sequence[float] | None = None,
        rates: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create z, H, R for common direct state measurements."""
        z_parts = []
        indices = []

        if position is not None:
            z_parts.append(np.asarray(position, dtype=float).reshape(3))
            indices.extend(range(0, 3))
        if velocity is not None:
            z_parts.append(np.asarray(velocity, dtype=float).reshape(3))
            indices.extend(range(3, 6))
        if attitude is not None:
            z_parts.append(np.asarray(attitude, dtype=float).reshape(3))
            indices.extend(range(6, 9))
        if rates is not None:
            z_parts.append(np.asarray(rates, dtype=float).reshape(3))
            indices.extend(range(9, 12))

        if not z_parts:
            raise ValueError("at least one measurement block must be provided")

        z = np.concatenate(z_parts)
        H = self.measurement_matrix(indices)
        R = self.default_measurement_noise(indices)
        return z, H, R

    def innovation(self, measurement: np.ndarray, H: np.ndarray) -> np.ndarray:
        innovation = np.asarray(measurement, dtype=float) - H @ self.x

        # Wrap angle innovations if the measurement includes roll/pitch/yaw.
        for row in range(H.shape[0]):
            matching = np.flatnonzero(np.isclose(H[row], 1.0))
            if matching.size == 1 and 6 <= matching[0] <= 8:
                innovation[row] = self._wrap_angle(innovation[row])

        return innovation

    @staticmethod
    def mahalanobis_distance(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
        innovation = np.asarray(innovation, dtype=float).reshape(-1)
        innovation_cov = np.asarray(innovation_cov, dtype=float)
        return float(innovation @ np.linalg.solve(innovation_cov, innovation))

    def update(
        self,
        measurement: Sequence[float],
        H: np.ndarray,
        measurement_noise: np.ndarray | None = None,
        gate_threshold: float | None = None,
    ) -> bool:
        """Run the EKF measurement update with a linear measurement model."""
        measurement = np.asarray(measurement, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        if H.shape[1] != STATE_DIM:
            raise ValueError("H must have shape (m, 12)")
        if H.shape[0] != measurement.size:
            raise ValueError("measurement size must match H rows")

        if measurement_noise is None:
            if H.shape[0] > STATE_DIM:
                raise ValueError("custom measurement_noise required for m > 12")
            measurement_noise = self.default_measurement_noise(
                np.argmax(H, axis=1).tolist()
            )
        else:
            measurement_noise = np.asarray(measurement_noise, dtype=float)

        if measurement_noise.shape != (measurement.size, measurement.size):
            raise ValueError("measurement_noise must have shape (m, m)")

        y = self.innovation(measurement, H)
        S = H @ self.P @ H.T + measurement_noise
        self.last_innovation = y.copy()
        self.last_S = S.copy()
        self.last_NIS = float(y.T @ np.linalg.solve(S, y))
        self.last_meas_dim = int(measurement.size)
        self.last_mahalanobis = self.last_NIS
        self.last_gate_threshold = None if gate_threshold is None else float(gate_threshold)
        self.last_update_accepted = True
        if gate_threshold is not None and self.last_mahalanobis > float(gate_threshold):
            self.last_update_accepted = False
            return False

        PHt = self.P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T

        self.x = self._wrap_state_angles(self.x + K @ y)
        identity = np.eye(STATE_DIM, dtype=float)
        joseph = identity - K @ H
        self.P = joseph @ self.P @ joseph.T + K @ measurement_noise @ K.T
        self._stabilize_covariance()
        return True

    def step(
        self,
        omega: Sequence[float],
        measurement: Sequence[float] | None = None,
        H: np.ndarray | None = None,
        dt: float | None = None,
        process_noise: np.ndarray | None = None,
        measurement_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience wrapper for predict followed by an optional update."""
        self.predict(omega=omega, dt=dt, process_noise=process_noise)
        if measurement is not None:
            if H is None:
                raise ValueError("H is required when providing a measurement")
            self.update(
                measurement=measurement,
                H=H,
                measurement_noise=measurement_noise,
            )
        return self.x.copy()

    def as_dict(self) -> dict:
        return {
            "x": self.position,
            "v": self.velocity,
            "ang": self.attitude,
            "rate": self.body_rates,
            "state": self.x.copy(),
            "covariance": self.P.copy(),
        }

    def diagnostics(self) -> dict:
        return {
            "innovation": None if self.last_innovation is None else self.last_innovation.copy(),
            "S": None if self.last_S is None else self.last_S.copy(),
            "nis": self.last_NIS,
            "meas_dim": self.last_meas_dim,
            "mahalanobis": self.last_mahalanobis,
            "gate_threshold": self.last_gate_threshold,
            "update_accepted": self.last_update_accepted,
            "P_diag": np.diag(self.P).copy(),
            "P_trace": float(np.trace(self.P)),
        }

    def compute_nees(self, true_state: Sequence[float]) -> float:
        true_state = np.asarray(true_state, dtype=float).reshape(self.x.shape)
        error = true_state - self.x
        for idx in range(ANG_IDX.start, ANG_IDX.stop):
            error[idx] = self._wrap_angle(error[idx])
        return float(error.T @ np.linalg.solve(self.P, error))

    def decouple_all_groups(self):
        """Zeros out all cross-correlations: pos/vel <-> att/rates during dropout."""
        self.P[0:6, 6:12] = 0
        self.P[6:12, 0:6] = 0


class PhoenixEKFAdapter:
    """Bridge Phoenix sensor-noise measurements into the standalone EKF."""

    def __init__(
        self,
        dt: float = 0.002,
        ekf: QuadcopterEKF | None = None,
        sensor_noise: EKFSensorNoise | None = None,
        use_velocity_measurements: bool = True,
        use_attitude_measurements: bool = True,
        use_baro_measurements: bool = True,
    ):
        self.dt = float(dt)
        self.ekf = ekf or QuadcopterEKF(dt=self.dt)
        # Keep a dedicated noise instance so EKF bias/random-walk state is local.
        self.sensor_noise = sensor_noise or EKFSensorNoise(
            sample_turn_on_bias_once=True,
            gyro_turn_on_bias_sigma=0.0,
        )
        self.ekf.R_default = np.diag(
            _sensor_measurement_noise_diag(self.sensor_noise, self.dt)
        ) * _EKF_R_SCALE
        self.use_velocity_measurements = bool(use_velocity_measurements)
        self.use_attitude_measurements = bool(use_attitude_measurements)
        self.use_baro_measurements = bool(use_baro_measurements)
        # Dropout tracking for adaptive process noise
        self.dropout_time = 0.0
        self.base_Q = self.ekf.Q.copy()
        # configurable decoupling threshold (seconds). Set to None to disable.
        self.decouple_after_s = 2.0

    @staticmethod
    def _state_vector(
        position: Sequence[float],
        velocity: Sequence[float],
        attitude: Sequence[float],
        rates: Sequence[float],
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(position, dtype=float).reshape(3),
                np.asarray(velocity, dtype=float).reshape(3),
                np.asarray(attitude, dtype=float).reshape(3),
                np.asarray(rates, dtype=float).reshape(3),
            ]
        )

    def reset(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        attitude: Sequence[float],
        rates: Sequence[float],
        covariance: np.ndarray | None = None,
    ) -> None:
        self.sensor_noise.reset()
        self.dropout_time = 0.0
        self.ekf.reset(
            state=self._state_vector(position, velocity, attitude, rates),
            covariance=covariance,
        )

    def build_noisy_measurement(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        attitude: Sequence[float],
        rates: Sequence[float],
        *,
        acceleration: Sequence[float] | None = None,
        dropout_active: bool = False,
        baro_altitude: float | None = None,
        odom_position: Sequence[float] | None = None,
        odom_velocity: Sequence[float] | None = None,
        optical_flow_velocity_xy: Sequence[float] | None = None,
        dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        dt = self.dt if dt is None else float(dt)
        acceleration = (
            np.zeros(3, dtype=float)
            if acceleration is None
            else np.asarray(acceleration, dtype=float).reshape(3)
        )

        noisy_pos, noisy_vel, noisy_att, noisy_rates, _ = self.sensor_noise.add_noise(
            pos=np.asarray(position, dtype=float).reshape(3),
            vel=np.asarray(velocity, dtype=float).reshape(3),
            rot=np.asarray(attitude, dtype=float).reshape(3),
            omega=np.asarray(rates, dtype=float).reshape(3),
            acc=acceleration,
            dt=dt,
        )

        measurement_components = [self.ekf.build_gyro_measurement(noisy_rates)]
        if self.use_attitude_measurements:
            measurement_components.append(self.ekf.build_attitude_measurement(noisy_att))
        if self.use_baro_measurements:
            if baro_altitude is None:
                baro_altitude = self.sensor_noise.add_noise_to_baro(float(np.asarray(position, dtype=float).reshape(3)[2]))
            measurement_components.append(
                self.ekf.build_baro_measurement(
                    float(baro_altitude),
                    std=self.sensor_noise.baro_noise_std,
                )
            )
        if not dropout_active:
            measurement_components.append(self.ekf.build_gps_measurement(noisy_pos))
            if self.use_velocity_measurements and odom_velocity is None and optical_flow_velocity_xy is None:
                measurement_components.append(self.ekf.build_velocity_measurement(noisy_vel))
        if odom_position is not None or odom_velocity is not None:
            odom_pos = noisy_pos if odom_position is None else np.asarray(odom_position, dtype=float).reshape(3)
            odom_vel = noisy_vel if odom_velocity is None else np.asarray(odom_velocity, dtype=float).reshape(3)
            measurement_components.append(
                self.ekf.build_odom_measurement(
                    position=odom_pos,
                    velocity=odom_vel,
                )
            )
        elif optical_flow_velocity_xy is not None:
            measurement_components.append(
                self.ekf.build_optical_flow_measurement(optical_flow_velocity_xy)
            )

        measurement, H, R = self.ekf.stack_measurements(*measurement_components)
        noisy_state = {
            "position": noisy_pos,
            "velocity": noisy_vel,
            "attitude": noisy_att,
            "rates": noisy_rates,
            "baro_altitude": None if baro_altitude is None else float(baro_altitude),
            "odom_position": None if odom_position is None else np.asarray(odom_position, dtype=float).reshape(3).copy(),
            "odom_velocity": None if odom_velocity is None else np.asarray(odom_velocity, dtype=float).reshape(3).copy(),
            "optical_flow_velocity_xy": (
                None
                if optical_flow_velocity_xy is None
                else np.asarray(optical_flow_velocity_xy, dtype=float).reshape(2).copy()
            ),
            "dropout_duration_s": float(self.dropout_time),
            "dropout_active": bool(dropout_active),
        }
        return measurement, H, R, noisy_state

    def step(
        self,
        motor_omega: Sequence[float],
        position: Sequence[float],
        velocity: Sequence[float],
        attitude: Sequence[float],
        rates: Sequence[float],
        *,
        acceleration: Sequence[float] | None = None,
        dropout_active: bool = False,
        baro_altitude: float | None = None,
        odom_position: Sequence[float] | None = None,
        odom_velocity: Sequence[float] | None = None,
        optical_flow_velocity_xy: Sequence[float] | None = None,
        dt: float | None = None,
    ) -> dict:
        dt = self.dt if dt is None else float(dt)
        # Adaptive EKF prediction: scale process noise with dropout time.
        if dropout_active:
            # accumulate dropout time
            self.dropout_time += dt
            self.ekf.predict_dropout(
                omega=motor_omega,
                dt=dt,
                dropout_time=self.dropout_time,
            )
            # optional decoupling after prolonged dropout
            if self.decouple_after_s is not None and self.dropout_time > float(self.decouple_after_s):
                self.ekf.decouple_all_groups()
        else:
            # reset dropout timer
            if self.dropout_time != 0.0:
                self.dropout_time = 0.0
            self.ekf.predict(omega=motor_omega, dt=dt)

        measurement, H, R, noisy_state = self.build_noisy_measurement(
            position=position,
            velocity=velocity,
            attitude=attitude,
            rates=rates,
            acceleration=acceleration,
            dropout_active=dropout_active,
            baro_altitude=baro_altitude,
            odom_position=odom_position,
            odom_velocity=odom_velocity,
            optical_flow_velocity_xy=optical_flow_velocity_xy,
            dt=dt,
        )

        # perform measurement update if measurements present
        if measurement is not None and H is not None:
            self.ekf.update(measurement=measurement, H=H, measurement_noise=R)
        estimate = self.ekf.as_dict()
        estimate["measurement"] = noisy_state
        estimate["dropout_duration_s"] = float(self.dropout_time)
        return estimate


if __name__ == "__main__":
    ekf = QuadcopterEKF(dt=0.002)
    omega_hover = np.full(4, 700.0, dtype=float)

    for _ in range(10):
        ekf.predict(omega_hover)

    z, H, R = ekf.build_measurement(
        position=[0.0, 0.0, 1.0],
        attitude=[0.0, 0.0, 0.0],
        rates=[0.0, 0.0, 0.0],
    )
    estimate = ekf.update(z, H, R)

    np.set_printoptions(precision=4, suppress=True)
    print("EKF state estimate:")
    print(estimate)
