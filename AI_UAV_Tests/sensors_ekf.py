"""EKF-specific sensor-noise model.

This intentionally does not modify the shared Phoenix `SensorNoise` class.
It provides a safe place for EKF-only sensor experiments without touching the
shared simulator sensor stack.
"""

from math import exp

import numpy as np

from phoenix_drone_simulation.envs.sensors import SensorNoise


class EKFSensorNoise(SensorNoise):
    """SensorNoise variant used only by the EKF/validation path."""

    def __init__(self, *args, sample_turn_on_bias_once: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_turn_on_bias_once = bool(sample_turn_on_bias_once)
        self.gyro_turn_on_bias = np.zeros(3, dtype=float)
        self.reset()

    def reset(self) -> None:
        """Reset colored gyro terms for a new EKF episode."""
        self.gyro_bias = np.zeros(3, dtype=float)
        if self.bypass or not self.sample_turn_on_bias_once:
            self.gyro_turn_on_bias = np.zeros(3, dtype=float)
        else:
            self.gyro_turn_on_bias = (
                self.gyro_turn_on_bias_sigma * np.random.normal(0.0, 1.0, 3)
            )

    def add_noise_to_omega(self, omega, dt):
        """Match the original model, with optional one-time turn-on bias."""
        assert omega.shape == (3,)

        sigma_g_d = self.gyro_noise_density / (dt ** 0.5)
        sigma_b_g_d = (
            -((sigma_g_d ** 2) * (self.gyro_bias_correlation_time / 2.0))
            * (exp(-2.0 * dt / self.gyro_bias_correlation_time) - 1.0)
        ) ** 0.5
        pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * np.random.normal(
            0.0, 1.0, 3
        )
        if self.sample_turn_on_bias_once:
            turn_on_bias = self.gyro_turn_on_bias
        else:
            turn_on_bias = self.gyro_turn_on_bias_sigma * np.random.normal(0.0, 1.0, 3)
        return (
            omega
            + turn_on_bias
            + self.gyro_bias
            + self.gyro_random_walk * np.random.normal(0.0, 1.0, 3)
        )
