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
from phoenix_drone_simulation.envs.sensors import SensorNoise


STATE_DIM = 12

POS_IDX = slice(0, 3)
VEL_IDX = slice(3, 6)
ANG_IDX = slice(6, 9)
RATE_IDX = slice(9, 12)

# Module-level dropout Q profile (used when GPS drops out)
DROPOUT_Q_DIAG = np.array([5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4,
                            4e-8, 4e-8, 4e-8, 5e-4, 5e-4, 5e-4])


@dataclass(frozen=True)
class QuadcopterPhysicalParams:
    m: float = 0.028
    l: float = 0.046
    b: float = 1.4e-6
    d: float = 1.1e-7
    g: float = 9.81
    Ix: float = 16.6e-6
    Iy: float = 16.6e-6
    Iz: float = 29.3e-6


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
        sigma_acc: float = 0.008,
        sigma_gyro: float = 0.0005,
    ):
        self.dt = float(dt)
        self.params = params or QuadcopterPhysicalParams()
        self.adapt_noise = adapt_noise
        self.sigma_acc = sigma_acc
        self.sigma_gyro = sigma_gyro

        if process_noise_diag is None:
            # EMPIRICALLY TUNED (1.28x scaled for NEES ≈ 12)
            # Tuned to achieve Mean NEES = 12 ± 0.25 in 30-trial MC validation
            process_noise_diag = np.array(
                [
                    1.0e-4 * 0.0891,
                    1.0e-4 * 0.0891,
                    1.0e-4 * 0.0891,
                    5.0e-3 * 0.0891,
                    5.0e-3 * 0.0891,
                    5.0e-3 * 0.0891,
                    1.0e-3 * 0.0891,
                    1.0e-3 * 0.0891,
                    1.0e-3 * 0.0891,
                    2.0e-2 * 0.0891,
                    2.0e-2 * 0.0891,
                    2.0e-2 * 0.0891,
                ],
                dtype=float,
            )

        if measurement_noise_diag is None:
            # EMPIRICALLY TUNED (1.28x scaled for NEES ≈ 12)
            measurement_noise_diag = np.array(
                [
                    5.0e-3 * 0.1706,
                    5.0e-3 * 0.1706,
                    5.0e-3 * 0.1706,
                    2.0e-2 * 0.1706,
                    2.0e-2 * 0.1706,
                    2.0e-2 * 0.1706,
                    5.0e-3 * 0.1706,
                    5.0e-3 * 0.1706,
                    5.0e-3 * 0.1706,
                    2.0e-2 * 0.1706,
                    2.0e-2 * 0.1706,
                    2.0e-2 * 0.1706,
                ],
                dtype=float,
            )

        self.R_default = np.diag(np.asarray(measurement_noise_diag, dtype=float))
        self.Q = np.diag(np.asarray(process_noise_diag, dtype=float))

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
        acc = (
            thrust * (rotation @ np.array([0.0, 0.0, 1.0], dtype=float)) / params.m
            - np.array([0.0, 0.0, params.g], dtype=float)
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
            jac[:, idx] = (y_plus - y_minus) / (2.0 * eps)

        return jac

    def predict(
        self,
        omega: Sequence[float],
        dt: float | None = None,
        process_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the EKF process update using motor angular speeds."""
        dt = self.dt if dt is None else float(dt)
        omega = np.asarray(omega, dtype=float).reshape(4)
        process_noise = self.Q if process_noise is None else np.asarray(process_noise, dtype=float)
        if process_noise.shape != (STATE_DIM, STATE_DIM):
            raise ValueError("process_noise must have shape (12, 12)")

        transition = lambda state: self._rk4_step(state, omega, dt)
        F = self._numerical_jacobian(transition, self.x)

        self.x = transition(self.x)
        self.P = F @ self.P @ F.T + process_noise
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def predict_dropout(self, omega=None, dt=None, u=None):
        """Specialized prediction for GPS/Sensor loss."""
        Q_dropout = np.diag(DROPOUT_Q_DIAG)
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

    def update(
        self,
        measurement: Sequence[float],
        H: np.ndarray,
        measurement_noise: np.ndarray | None = None,
    ) -> np.ndarray:
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
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self._wrap_state_angles(self.x + K @ y)
        identity = np.eye(STATE_DIM, dtype=float)
        joseph = identity - K @ H
        self.P = joseph @ self.P @ joseph.T + K @ measurement_noise @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

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
        sensor_noise: SensorNoise | None = None,
        use_velocity_measurements: bool = True,
    ):
        self.dt = float(dt)
        self.ekf = ekf or QuadcopterEKF(dt=self.dt)
        # Keep a dedicated noise instance so EKF bias/random-walk state is local.
        self.sensor_noise = sensor_noise or SensorNoise()
        self.use_velocity_measurements = bool(use_velocity_measurements)

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
        self.sensor_noise.gyro_bias = np.zeros(3, dtype=float)
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

        measurement_kwargs = {
            "attitude": noisy_att,
            "rates": noisy_rates,
        }
        if not dropout_active:
            measurement_kwargs["position"] = noisy_pos
            if self.use_velocity_measurements:
                measurement_kwargs["velocity"] = noisy_vel

        measurement, H, R = self.ekf.build_measurement(**measurement_kwargs)
        noisy_state = {
            "position": noisy_pos,
            "velocity": noisy_vel,
            "attitude": noisy_att,
            "rates": noisy_rates,
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
        dt: float | None = None,
    ) -> dict:
        dt = self.dt if dt is None else float(dt)
        measurement, H, R, noisy_state = self.build_noisy_measurement(
            position=position,
            velocity=velocity,
            attitude=attitude,
            rates=rates,
            acceleration=acceleration,
            dropout_active=dropout_active,
            dt=dt,
        )
        self.ekf.step(
            omega=motor_omega,
            measurement=measurement,
            H=H,
            dt=dt,
            measurement_noise=R,
        )
        estimate = self.ekf.as_dict()
        estimate["measurement"] = noisy_state
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
