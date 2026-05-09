"""Shared LSTM drift-prediction helpers for the 12-state QuadcopterEKF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from AI_UAV_Tests.core.trajectories_library import FlightMission, Trajectories


SHADOW_RESET_INTERVAL = 2500
HORIZON_STRIDE = 100
NUM_HORIZONS = 7
BUFFER_LEN = 300
TARGET_SPEED = 1.2
TAKEOFF_Z = 0.16
MAX_WAYPOINT_SHIFT = 0.08
PRECOMP_LEAD_TIME_S = 1.0
PSEUDO_VEL_STD_XY = 0.10
PSEUDO_VEL_STD_Z = 0.08
PSEUDO_POS_STD_XY = 0.08
PSEUDO_POS_STD_Z = 0.05
PRECOMP_CORRECTION_GAIN = 0.35
MIN_REF_SPEED_FOR_CORRECTION = 0.15
MOVING_PHASE_NAMES = frozenset({"circle", "square", "helix", "sine", "figure_eight"})
STATE_CORRECTION_LEAD_TIME_S = 0.5
STATE_CORRECTION_GAIN_XY = 0.75
STATE_CORRECTION_GAIN_Z = 0.40
STATE_CORRECTION_MAX_XY = 0.35
STATE_CORRECTION_MAX_Z = 0.10
STATE_CORRECTION_WARMUP_S = 0.75

STANDARD_TRAJECTORIES = (
    "hover",
    "circle",
    "square",
    "helix",
    "sine",
    "figure_eight",
)
STANDARD_TRAJECTORY_WEIGHTS = (
    0.15,
    0.22,
    0.18,
    0.15,
    0.15,
    0.15,
)

DEFAULT_DATA_DIR = Path("data/quadcopter_ekf_drift")
DEFAULT_MODEL_PATH = Path("experiments/quadcopter_ekf_drift_lstm.pt")


def _rng_uniform(rng, low: float, high: float) -> float:
    if rng is None:
        return 0.5 * (float(low) + float(high))
    return float(rng.uniform(float(low), float(high)))


class QuadcopterPositionDriftLSTM(nn.Module):
    """Multi-horizon LSTM drift predictor using EKF velocity + gyro + t_norm."""

    def __init__(self, horizon_steps: int = NUM_HORIZONS, hidden_size: int = 256):
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        self.proj = nn.Linear(7, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 3),
                )
                for _ in range(self.horizon_steps)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        return torch.stack([head(final) for head in self.heads], dim=1)


class QuadcopterDriftDataset(Dataset):
    """Sequence dataset for multi-horizon EKF drift prediction."""

    def __init__(
        self,
        missions: list[dict],
        seq_len: int = BUFFER_LEN,
        stride: int = 10,
        horizon_stride: int = HORIZON_STRIDE,
        num_horizons: int = NUM_HORIZONS,
        shadow_reset_interval: int = SHADOW_RESET_INTERVAL,
    ):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.seq_len = int(seq_len)

        ri = int(shadow_reset_interval)
        hs = int(horizon_stride)
        nh = int(num_horizons)

        for mission in missions:
            features = np.asarray(mission["features"], dtype=np.float32)
            dr_pos = np.asarray(mission["dr_pos"], dtype=np.float32)
            true_pos = np.asarray(mission["true_pos"], dtype=np.float32)
            drift = dr_pos - true_pos
            total_steps = int(features.shape[0])

            for start in range(0, total_steps, int(stride)):
                end = start + self.seq_len
                horizon_idxs = [end + hs * (h + 1) for h in range(nh)]
                if horizon_idxs[-1] >= total_steps:
                    break
                if end // ri != start // ri:
                    continue
                if any(idx // ri != (end - 1) // ri for idx in horizon_idxs):
                    continue

                seq = features[start:end]
                tgts = np.array([drift[idx] for idx in horizon_idxs], dtype=np.float32)
                self.samples.append((seq, tgts))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, tgts = self.samples[idx]
        return torch.from_numpy(seq).float(), torch.from_numpy(tgts).float()


def build_standard_mission(
    trajectory: str,
    *,
    mission_z: float = 1.0,
) -> FlightMission:
    mission = FlightMission(default_z=mission_z, ground_z=0.0)
    trajectory = str(trajectory)

    if trajectory == "hover":
        mission.add_takeoff(duration=3.0, target_z=mission_z)
        mission.add_hover(duration=16.0, z=mission_z)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    if trajectory == "circle":
        mission.add_takeoff(duration=3.0, target_z=mission_z)
        mission.add_hover(duration=1.0, z=mission_z)
        mission.add_circle(
            duration=12.0,
            radius=1.0,
            period=12.0,
            z=mission_z,
            center_xy=(-1.0, 0.0),
        )
        mission.add_hover(duration=2.0, z=mission_z)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    if trajectory == "square":
        mission.add_takeoff(duration=3.0, target_z=mission_z)
        mission.add_hover(duration=1.0, z=mission_z)
        mission.add_square(
            duration=16.0,
            side=1.5,
            period=16.0,
            z=mission_z,
            offset_xy=(0.0, 0.0),
        )
        mission.add_hover(duration=2.0, z=mission_z)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    if trajectory == "figure_eight":
        mission.add_takeoff(duration=3.0, target_z=mission_z)
        mission.add_hover(duration=1.0, z=mission_z)
        mission.add_figure_eight(
            duration=14.0,
            scale=0.8,
            period=14.0,
            z=mission_z,
            center_xy=(0.0, 0.0),
        )
        mission.add_hover(duration=2.0, z=mission_z)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    if trajectory == "helix":
        mission.add_takeoff(duration=3.0, target_z=mission_z)
        mission.add_hover(duration=1.0, z=mission_z)

        def _helix(local_t: float):
            pos, vel = Trajectories.helix_traj(local_t, radius=0.7, height=0.6, period=14.0)
            pos = pos.copy()
            pos[2] += mission_z
            return pos, vel

        mission.add_phase("helix", 14.0, _helix)
        mission.add_hover(duration=2.0, z=mission_z + 0.6)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    if trajectory == "sine":
        mission.add_takeoff(duration=3.0, target_z=mission_z)
        mission.add_hover(duration=1.0, z=mission_z)

        def _sine(local_t: float):
            pos, vel = Trajectories.sine_traj(local_t, amplitude=0.6, freq=0.12, z=mission_z)
            return pos, vel

        mission.add_phase("sine", 14.0, _sine)
        mission.add_hover(duration=2.0, z=mission_z)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    raise ValueError(f"Unknown standard trajectory: {trajectory}")


def build_randomized_standard_mission(
    trajectory: str,
    *,
    rng=None,
) -> FlightMission:
    """Build a shorter randomized mission for faster task-specific LSTM collection."""
    trajectory = str(trajectory)
    mission_z = _rng_uniform(rng, 0.95, 1.05)
    mission = FlightMission(default_z=mission_z, ground_z=0.0)

    takeoff_duration = _rng_uniform(rng, 2.4, 3.0)
    landing_duration = _rng_uniform(rng, 4.0, 5.0)
    pre_hover = _rng_uniform(rng, 0.5, 1.0)
    post_hover = _rng_uniform(rng, 0.6, 1.2)

    if trajectory == "hover":
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_hover(duration=_rng_uniform(rng, 8.0, 11.0), z=mission_z)
        mission.add_landing(duration=landing_duration, ground_z=0.055)
        return mission

    if trajectory == "circle":
        radius = _rng_uniform(rng, 0.70, 1.10)
        period = _rng_uniform(rng, 8.0, 10.5)
        duration = _rng_uniform(rng, 7.5, 9.5)
        center_x = -radius + _rng_uniform(rng, -0.15, 0.15)
        center_y = _rng_uniform(rng, -0.20, 0.20)
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_hover(duration=pre_hover, z=mission_z)
        mission.add_circle(
            duration=duration,
            radius=radius,
            period=period,
            z=mission_z,
            center_xy=(center_x, center_y),
        )
        mission.add_hover(duration=post_hover, z=mission_z)
        mission.add_landing(duration=landing_duration, ground_z=0.055)
        return mission

    if trajectory == "square":
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_hover(duration=pre_hover, z=mission_z)
        mission.add_square(
            duration=_rng_uniform(rng, 10.0, 13.0),
            side=_rng_uniform(rng, 1.1, 1.7),
            period=_rng_uniform(rng, 10.0, 13.0),
            z=mission_z,
            offset_xy=(_rng_uniform(rng, -0.15, 0.15), _rng_uniform(rng, -0.15, 0.15)),
        )
        mission.add_hover(duration=post_hover, z=mission_z)
        mission.add_landing(duration=landing_duration, ground_z=0.055)
        return mission

    if trajectory == "figure_eight":
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_hover(duration=pre_hover, z=mission_z)
        mission.add_figure_eight(
            duration=_rng_uniform(rng, 10.0, 12.5),
            scale=_rng_uniform(rng, 0.65, 0.95),
            period=_rng_uniform(rng, 10.0, 12.5),
            z=mission_z,
            center_xy=(_rng_uniform(rng, -0.10, 0.10), _rng_uniform(rng, -0.10, 0.10)),
        )
        mission.add_hover(duration=post_hover, z=mission_z)
        mission.add_landing(duration=landing_duration, ground_z=0.055)
        return mission

    if trajectory == "helix":
        radius = _rng_uniform(rng, 0.55, 0.80)
        height = _rng_uniform(rng, 0.40, 0.70)
        period = _rng_uniform(rng, 10.0, 13.0)
        duration = _rng_uniform(rng, 10.0, 12.5)
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_hover(duration=pre_hover, z=mission_z)

        def _helix(local_t: float):
            pos, vel = Trajectories.helix_traj(local_t, radius=radius, height=height, period=period)
            pos = pos.copy()
            pos[2] += mission_z
            return pos, vel

        mission.add_phase("helix", duration, _helix)
        mission.add_hover(duration=post_hover, z=mission_z + height)
        mission.add_landing(duration=landing_duration, ground_z=0.055)
        return mission

    if trajectory == "sine":
        amplitude = _rng_uniform(rng, 0.45, 0.75)
        freq = _rng_uniform(rng, 0.10, 0.16)
        duration = _rng_uniform(rng, 10.0, 12.5)
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_hover(duration=pre_hover, z=mission_z)

        def _sine(local_t: float):
            pos, vel = Trajectories.sine_traj(local_t, amplitude=amplitude, freq=freq, z=mission_z)
            return pos, vel

        mission.add_phase("sine", duration, _sine)
        mission.add_hover(duration=post_hover, z=mission_z)
        mission.add_landing(duration=landing_duration, ground_z=0.055)
        return mission

    raise ValueError(f"Unknown standard trajectory: {trajectory}")


def thrust_to_action(U1: float, mass: float, g: float = 9.81) -> float:
    hover_T = mass * g
    return float(np.clip((U1 / hover_T - 0.9) / 0.4, -1.0, 1.0))


def motor_omega_from_forces(
    motor_forces: np.ndarray,
    thrust_coeff: float,
) -> np.ndarray:
    motor_forces = np.asarray(motor_forces, dtype=float).reshape(4)
    return np.sqrt(np.clip(motor_forces, 0.0, np.inf) / float(thrust_coeff))


def build_dropout_windows(
    total_time: float,
    *,
    spacing_s: float = 10.0,
    duration_s: float = 3.5,
    start_pad_s: float = 3.0,
    end_pad_s: float = 1.0,
) -> list[tuple[float, float]]:
    num_dropouts = int(total_time / spacing_s)
    windows: list[tuple[float, float]] = []
    for i in range(num_dropouts):
        start = start_pad_s + i * (total_time - (start_pad_s + end_pad_s)) / max(num_dropouts, 1)
        end = start + duration_s
        if end < total_time - end_pad_s:
            windows.append((float(start), float(end)))
    return windows


def build_phase_dropout_windows(
    mission: FlightMission,
    *,
    duration_s: float = 3.0,
    margin_s: float = 1.0,
    max_windows: int = 2,
    include_hover: bool = False,
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    valid_names = set(MOVING_PHASE_NAMES)
    if include_hover:
        valid_names.add("hover")

    for phase in mission.phases:
        phase_name = str(phase.name)
        if phase_name not in valid_names:
            continue
        phase_duration = float(phase.t_end - phase.t_start)
        usable = phase_duration - 2.0 * float(margin_s)
        if usable < float(duration_s):
            continue
        start = float(phase.t_start + margin_s + 0.5 * (usable - float(duration_s)))
        end = start + float(duration_s)
        windows.append((start, end))
        if len(windows) >= int(max_windows):
            break
    return windows


def build_lstm_feature(
    ekf_velocity: np.ndarray,
    gyro_rates: np.ndarray,
    t_norm: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(ekf_velocity, dtype=np.float32).reshape(3),
            np.asarray(gyro_rates, dtype=np.float32).reshape(3),
            np.array([np.float32(t_norm)], dtype=np.float32),
        ]
    )


def run_reference_velocity_dropout_step(
    adapter,
    *,
    motor_forces: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    attitude: np.ndarray,
    rates: np.ndarray,
    velocity_ref: np.ndarray,
    position_ref: np.ndarray | None = None,
    dt: float,
    acceleration: np.ndarray | None = None,
) -> dict:
    dt = float(dt)
    adapter.dropout_time += dt
    omega_predict = motor_omega_from_forces(motor_forces, adapter.ekf.params.b)
    adapter.ekf.predict_dropout(
        omega=omega_predict,
        dt=dt,
        dropout_time=adapter.dropout_time,
    )
    if adapter.decouple_after_s is not None and adapter.dropout_time > float(adapter.decouple_after_s):
        adapter.ekf.decouple_all_groups()

    acceleration = (
        np.zeros(3, dtype=float)
        if acceleration is None
        else np.asarray(acceleration, dtype=float).reshape(3)
    )
    noisy_pos, noisy_vel, noisy_att, noisy_rates, _ = adapter.sensor_noise.add_noise(
        np.asarray(position, dtype=float).reshape(3),
        np.asarray(velocity, dtype=float).reshape(3),
        np.asarray(attitude, dtype=float).reshape(3),
        np.asarray(rates, dtype=float).reshape(3),
        acceleration,
        dt,
    )
    baro_z = adapter.sensor_noise.add_noise_to_baro(float(position[2]))
    measurement_components = [
        adapter.ekf.build_attitude_measurement(noisy_att),
        adapter.ekf.build_gyro_measurement(noisy_rates),
        adapter.ekf.build_velocity_pseudo_measurement(
            velocity_ref,
            std_xy=PSEUDO_VEL_STD_XY,
            std_z=PSEUDO_VEL_STD_Z,
        ),
        adapter.ekf.build_baro_measurement(
            baro_z,
            std=adapter.sensor_noise.baro_noise_std,
        ),
    ]
    if position_ref is not None:
        measurement_components.append(
            adapter.ekf.build_position_pseudo_measurement(
                position_ref,
                std_xy=PSEUDO_POS_STD_XY,
                std_z=PSEUDO_POS_STD_Z,
            )
        )
    measurement, H, R = adapter.ekf.stack_measurements(*measurement_components)
    adapter.ekf.update(measurement=measurement, H=H, measurement_noise=R)
    estimate = adapter.ekf.as_dict()
    estimate["measurement"] = {
        "position": noisy_pos,
        "velocity": noisy_vel,
        "attitude": noisy_att,
        "rates": noisy_rates,
        "baro_z": float(baro_z),
        "velocity_ref": np.asarray(velocity_ref, dtype=float).reshape(3).copy(),
        "dropout_active": True,
    }
    return estimate


def apply_lstm_reference_precomp(
    *,
    ref_pos: np.ndarray,
    lstm_preds: np.ndarray,
    ref_vel: np.ndarray | None = None,
    dt: float,
    lead_time_s: float = PRECOMP_LEAD_TIME_S,
    horizon_stride: int = HORIZON_STRIDE,
    max_waypoint_shift: float = MAX_WAYPOINT_SHIFT,
    correction_gain: float = PRECOMP_CORRECTION_GAIN,
    min_ref_speed: float = MIN_REF_SPEED_FOR_CORRECTION,
) -> np.ndarray:
    adjusted = np.asarray(ref_pos, dtype=float).reshape(3).copy()
    lstm_preds = np.asarray(lstm_preds, dtype=float)
    h_idx = min(
        int(float(lead_time_s) / (float(horizon_stride) * float(dt))),
        lstm_preds.shape[0] - 1,
    )
    drift = lstm_preds[h_idx].copy()
    drift[2] = 0.0

    if ref_vel is not None:
        ref_vel = np.asarray(ref_vel, dtype=float).reshape(3)
        vel_norm = float(np.linalg.norm(ref_vel[:2]))
        if vel_norm < float(min_ref_speed):
            return adjusted
        if vel_norm > 1.0e-3:
            u = ref_vel[:2] / vel_norm
            shift_xy = np.dot(drift[:2], u) * u
        else:
            shift_xy = drift[:2]
    else:
        shift_xy = drift[:2]

    shift_xy *= float(correction_gain)
    shift_mag = float(np.linalg.norm(shift_xy))
    if shift_mag > float(max_waypoint_shift):
        shift_xy *= float(max_waypoint_shift) / shift_mag

    adjusted[0] += shift_xy[0]
    adjusted[1] += shift_xy[1]
    return adjusted


def compute_lstm_state_correction(
    *,
    lstm_preds: np.ndarray,
    dt: float,
    ref_vel: np.ndarray | None = None,
    dropout_time_s: float = 0.0,
    lead_time_s: float = STATE_CORRECTION_LEAD_TIME_S,
    horizon_stride: int = HORIZON_STRIDE,
    gain_xy: float = STATE_CORRECTION_GAIN_XY,
    gain_z: float = STATE_CORRECTION_GAIN_Z,
    max_xy: float = STATE_CORRECTION_MAX_XY,
    max_z: float = STATE_CORRECTION_MAX_Z,
    min_ref_speed: float = MIN_REF_SPEED_FOR_CORRECTION,
    warmup_s: float = STATE_CORRECTION_WARMUP_S,
) -> np.ndarray:
    lstm_preds = np.asarray(lstm_preds, dtype=float)
    correction = np.zeros(3, dtype=float)
    if lstm_preds.size == 0:
        return correction

    h_idx = min(
        int(float(lead_time_s) / (float(horizon_stride) * float(dt))),
        lstm_preds.shape[0] - 1,
    )
    drift = np.asarray(lstm_preds[h_idx], dtype=float).reshape(3)

    if ref_vel is not None:
        ref_vel = np.asarray(ref_vel, dtype=float).reshape(3)
        if np.linalg.norm(ref_vel[:2]) < float(min_ref_speed):
            drift[:2] = 0.0

    correction[:2] = float(gain_xy) * drift[:2]
    correction[2] = float(gain_z) * drift[2]

    xy_norm = float(np.linalg.norm(correction[:2]))
    if xy_norm > float(max_xy):
        correction[:2] *= float(max_xy) / xy_norm
    correction[2] = float(np.clip(correction[2], -float(max_z), float(max_z)))

    warmup = 1.0
    if float(warmup_s) > 1.0e-6:
        warmup = np.clip(float(dropout_time_s) / float(warmup_s), 0.0, 1.0)
    correction *= float(warmup)
    return correction
