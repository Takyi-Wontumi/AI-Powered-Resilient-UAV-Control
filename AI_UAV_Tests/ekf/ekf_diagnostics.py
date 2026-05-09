"""Diagnostics utilities for EKF consistency and IMU noise analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class InnovationCovarianceStats:
    """Accumulate innovation second moments against predicted S."""

    dim: int
    label: str

    def __post_init__(self) -> None:
        self.count = 0
        self.outer_sum = np.zeros((self.dim, self.dim), dtype=float)
        self.predicted_s_sum = np.zeros((self.dim, self.dim), dtype=float)

    def add(self, innovation: np.ndarray, predicted_s: np.ndarray) -> None:
        innovation = np.asarray(innovation, dtype=float).reshape(self.dim)
        predicted_s = np.asarray(predicted_s, dtype=float).reshape(self.dim, self.dim)
        self.outer_sum += np.outer(innovation, innovation)
        self.predicted_s_sum += predicted_s
        self.count += 1

    @property
    def empirical_cov(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((self.dim, self.dim), dtype=float)
        return self.outer_sum / float(self.count)

    @property
    def predicted_cov(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((self.dim, self.dim), dtype=float)
        return self.predicted_s_sum / float(self.count)

    def diag_ratio(self) -> np.ndarray:
        predicted_diag = np.maximum(np.diag(self.predicted_cov), 1.0e-15)
        return np.diag(self.empirical_cov) / predicted_diag

    def frobenius_relative_error(self) -> float:
        denom = float(np.linalg.norm(self.predicted_cov, ord="fro"))
        if denom <= 1.0e-15:
            return 0.0
        return float(
            np.linalg.norm(self.empirical_cov - self.predicted_cov, ord="fro") / denom
        )

    def summary_lines(self) -> list[str]:
        lines = [f"{self.label}: samples={self.count} dim={self.dim}"]
        if self.count == 0:
            lines.append("  no innovation samples collected")
            return lines
        lines.append(
            "  diag empirical/predicted ratio = "
            + np.array2string(self.diag_ratio(), precision=3, separator=", ")
        )
        lines.append(
            f"  relative Frobenius error       = {self.frobenius_relative_error():.3f}"
        )
        lines.append(
            "  empirical diag                = "
            + np.array2string(np.diag(self.empirical_cov), precision=5, separator=", ")
        )
        lines.append(
            "  predicted diag                = "
            + np.array2string(np.diag(self.predicted_cov), precision=5, separator=", ")
        )
        return lines


def write_innovation_report(
    stats: list[InnovationCovarianceStats],
    *,
    save_dir: str | None = None,
    file_name: str = "innovation_covariance_report.txt",
) -> str:
    lines = ["Innovation covariance matching report"]
    for item in stats:
        lines.extend(item.summary_lines())
    text = "\n".join(lines) + "\n"
    print(text)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / file_name).write_text(text, encoding="utf-8")
    return text


def allan_deviation(samples: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute Allan deviation for a 1D sample sequence."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    n = samples.size
    if n < 4:
        raise ValueError("need at least 4 samples for Allan deviation")

    max_m = max(2, n // 4)
    m_values = np.unique(
        np.clip(
            np.logspace(0, np.log10(max_m), num=min(40, max_m), base=10.0).astype(int),
            1,
            max_m,
        )
    )
    taus = []
    adevs = []
    for m in m_values:
        cluster_count = n // m
        if cluster_count < 3:
            continue
        trimmed = samples[: cluster_count * m]
        clusters = trimmed.reshape(cluster_count, m).mean(axis=1)
        diff = np.diff(clusters)
        allan_var = 0.5 * np.mean(diff ** 2)
        taus.append(m * dt)
        adevs.append(np.sqrt(allan_var))
    return np.asarray(taus, dtype=float), np.asarray(adevs, dtype=float)


def _fit_loglog_slope(
    taus: np.ndarray,
    adev: np.ndarray,
    slope_target: float,
    slope_tol: float = 0.2,
) -> tuple[float, float] | None:
    """Fit one Allan-deviation region near the requested log-log slope."""
    taus = np.asarray(taus, dtype=float)
    adev = np.asarray(adev, dtype=float)
    if taus.size < 3:
        return None
    log_tau = np.log10(taus)
    log_adev = np.log10(np.maximum(adev, 1.0e-15))
    local_slopes = np.diff(log_adev) / np.diff(log_tau)
    mask = np.abs(local_slopes - slope_target) <= slope_tol
    if not np.any(mask):
        return None
    idx = np.flatnonzero(mask)
    x = log_tau[idx]
    y = log_adev[idx]
    coeff = np.polyfit(x, y, 1)
    return float(coeff[0]), float(coeff[1])


def estimate_allan_noise_parameters(
    samples: np.ndarray,
    dt: float,
) -> dict[str, float | np.ndarray]:
    """Estimate white-noise and random-walk coefficients from Allan deviation."""
    taus, adev = allan_deviation(samples, dt)
    result: dict[str, float | np.ndarray] = {
        "taus": taus,
        "adev": adev,
    }

    white_fit = _fit_loglog_slope(taus, adev, slope_target=-0.5)
    if white_fit is not None:
        _, intercept = white_fit
        # sigma(τ) = N / sqrt(τ)  -> log10 sigma = log10 N - 0.5 log10 τ
        result["white_noise_coeff"] = 10.0 ** intercept

    rw_fit = _fit_loglog_slope(taus, adev, slope_target=0.5)
    if rw_fit is not None:
        _, intercept = rw_fit
        # sigma(τ) = K * sqrt(τ / 3) -> K = sigma * sqrt(3 / τ)
        result["random_walk_coeff"] = np.sqrt(3.0) * (10.0 ** intercept)

    return result
