#!/usr/bin/env python3
"""
UAVPrecompEnv — gymnasium.Env wrapping the full EKF + frozen LSTM + PID UAV mission.
======================================================================================
Used for PPO training of a residual waypoint correction policy.

Observation (42-dim):
  [0:21]  lstm_preds_flat  — 7-horizon drift predictions (7×3), clipped ±1 m
  [21]    t_norm           — steps_since_dropout / SHADOW_RESET_INTERVAL (0→1)
  [22:25] vel              — world-frame velocity (m/s), clipped ±5
  [25:27] wp_dir_xy        — normalised unit vector toward current waypoint (XY)
  [27]    dist_to_wp       — distance to current waypoint (m), clipped [0, 5]
  [28]    gps_ok           — 1.0 if GPS active, 0.0 during dropout
  [29:32] ekf_pos_std      — EKF XYZ position std-dev (m), clipped [0, 5]; grows during dropout
  [32:34] applied_alpha_xy — current α applied per axis (centred: α−1), clipped ±1.0
  [34:36] v0_xy            — normalised direction: wp[i] → wp[i+1]
  [36:38] v1_xy            — normalised direction: wp[i+1] → wp[i+2]
  [38:40] v2_xy            — normalised direction: wp[i+2] → wp[i+3]
  [40:42] seg_lengths      — [dist(wp[i]→wp[i+1]), dist(wp[i+1]→wp[i+2])], clipped [0, 5]

Action (2-dim, [α_x, α_y]):
  α_x, α_y ∈ [0.0, 2.0] — per-axis confidence multipliers on LSTM drift vector.
  adjusted_wp = waypoint + clamp([α_x×drift_x, α_y×drift_y], MAX_WAYPOINT_SHIFT)
  α=1.0 → pure LSTM, α<1 → attenuate that axis, α>1 → amplify that axis.
  Waypoint displacement is hard-capped at MAX_WAYPOINT_SHIFT=0.25 m to prevent
  extreme shifts from pushing waypoints outside reachable range.

Reward (V8 — true counterfactual):
  Dense  (dropout only, after buffer fills):
    ekf_drift = ekf_pos[:2] − true_pos[:2]          (actual XY EKF drift)
    dp        = lstm_preds[h_idx][:2]                (LSTM XY drift prediction)
    lstm_err  = |dp − ekf_drift|                     (LSTM arrival error)
    rl_err    = |[α_x×dp_x, α_y×dp_y] − ekf_drift|  (RL arrival error)
    +2.0 × (lstm_err − rl_err)    positive when RL beats LSTM-only baseline
  Arrival:               1 − min(dist(true_pos[:2], intended_wp[:2]), 0.5) / 0.5
  Regularisation:        −0.001 × ((α_x − 1)² + (α_y − 1)²)  (tie-breaker only)
"""

import sys
import os

_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, 'AI_UAV_Tests'))
sys.path.insert(0, os.path.join(_DIR, 'GPS_Dropout_Recovery'))

import numpy as np
import torch
import torch.nn as nn
import gymnasium
from gymnasium import spaces
from pathlib import Path

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import QuadcopterEKF
from AI_UAV_Tests.sensors_ekf import EKFSensorNoise

# ── Constants (must match data collector / precomp trainer) ───────────────────
SHADOW_RESET_INTERVAL = 2500    # steps between EKF resets in collector
HORIZON_STRIDE        = 100     # steps per LSTM horizon (= 0.5 s at 200 Hz)
NUM_HORIZONS          = 7       # +0.5 s … +3.5 s
WAYPOINT_TOLERANCE    = 0.5     # metres — arrival radius
TARGET_SPEED          = 1.2     # m/s — used for horizon selection
TAKEOFF_Z             = 0.16    # metres — takeoff target altitude
PATHFOLLOW_TRIGGER_Z  = 0.15   # metres — altitude at which path-following activates
BUFFER_LEN            = 300     # IMU history length (same as training)
MAX_WAYPOINT_SHIFT    = 0.25    # metres — hard cap on adjusted waypoint displacement (V11)

LSTM_MODEL_PATH = Path(_DIR) / 'experiments' / 'lstm_position_drift_precomp.pt'
PSEUDO_VEL_STD_XY = 2.10
PSEUDO_VEL_STD_Z = 4.00
EKF_SIGMA_GYRO_BIAS = 0.001
EKF_Q_BIAS_FLOOR = 1.0e-8
EKF_INITIAL_GYRO_BIAS_STD = 0.001


# ── LSTM architecture (must mirror train_lstm_position_drift_precomp.py) ──────
class PositionDriftLSTMPrecomp(nn.Module):
    """7-input (vel+gyro+t_norm), 7-horizon frozen LSTM for waypoint pre-compensation."""
    def __init__(self, horizon_steps=7, hidden_size=256):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.proj = nn.Linear(7, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3)
            )
            for _ in range(horizon_steps)
        ])

    def forward(self, x):
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        return torch.stack([h(final) for h in self.heads], dim=1)  # (B, 7, 3)


def _true_gyro_bias_from_sensor_noise(sensor_noise):
    turn_on_bias = np.asarray(
        getattr(sensor_noise, "gyro_turn_on_bias", np.zeros(3, dtype=float)),
        dtype=float,
    ).reshape(3)
    colored_bias = np.asarray(
        getattr(sensor_noise, "gyro_bias", np.zeros(3, dtype=float)),
        dtype=float,
    ).reshape(3)
    return turn_on_bias + colored_bias


def _motor_omega_from_applied_forces(env, ekf_model: QuadcopterEKF) -> np.ndarray:
    motor_forces = np.asarray(env.drone.y, dtype=float).reshape(4)
    thrust_coeff = float(ekf_model.params.b)
    return np.sqrt(np.clip(motor_forces, 0.0, np.inf) / thrust_coeff)


# ── Search pattern generator (mirrors demo_adaptive_search_v3_hybrid_precomp) ─
class _PatternGenerator:
    def __init__(self, area_size=(3.5, 2.5), altitude=1.0):
        self.x_min, self.x_max = -area_size[0] / 2, area_size[0] / 2
        self.y_min, self.y_max = -area_size[1] / 2, area_size[1] / 2
        self.z = altitude
        self.area_size = area_size

    def zigzag_search(self, num_passes=8, randomness=0.3):
        waypoints = []
        x_width   = self.x_max - self.x_min
        y_spacing = (self.y_max - self.y_min) / (num_passes + 1)
        for i in range(num_passes):
            y = self.y_min + (i + 1) * y_spacing
            x_start = np.clip(self.x_min + np.random.uniform(-randomness, randomness) * x_width * 0.05,
                              self.x_min, self.x_max)
            x_end   = np.clip(self.x_max + np.random.uniform(-randomness, randomness) * x_width * 0.05,
                              self.x_min, self.x_max)
            for j in range(3):
                x   = x_start + (x_end - x_start) * j / 2
                y_v = np.clip(y + np.random.uniform(-randomness, randomness) * y_spacing * 0.3,
                              self.y_min, self.y_max)
                waypoints.append(np.array([x, y_v, self.z]))
            if i < num_passes - 1:
                y_turn = np.clip(y + y_spacing * 0.5, self.y_min, self.y_max)
                waypoints.append(np.array([x_end, y_turn, self.z]))
        return np.array(waypoints)

    def spiral_search(self, rotations=4, randomness=0.2):
        waypoints = []
        n = 40
        for i in range(n):
            t = 2 * np.pi * rotations * i / n
            r = (self.area_size[0] / 2) * i / n
            x = np.clip(r * np.cos(t) + np.random.uniform(-randomness, randomness) * self.area_size[0] * 0.1,
                        self.x_min, self.x_max)
            y = np.clip((r / 1.5) * np.sin(t) + np.random.uniform(-randomness, randomness) * self.area_size[1] * 0.1,
                        self.y_min, self.y_max)
            waypoints.append(np.array([x, y, self.z]))
        return np.array(waypoints)

    def perimeter_search(self, num_laps=3, randomness=0.15):
        waypoints = []
        perimeter = 2 * (self.x_max - self.x_min) + 2 * (self.y_max - self.y_min)
        pts_per_lap = max(8, int(round(perimeter * 2.4)))
        for i in range(pts_per_lap * num_laps):
            frac = (i % pts_per_lap) / pts_per_lap
            p    = frac * perimeter
            if p < (self.x_max - self.x_min):
                x, y = self.x_min + p, self.y_min
            elif p < 2 * (self.x_max - self.x_min):
                x, y = self.x_max, self.y_min + (p - (self.x_max - self.x_min))
            elif p < 2 * (self.x_max - self.x_min) + (self.y_max - self.y_min):
                x, y = self.x_max - (p - 2 * (self.x_max - self.x_min)), self.y_max
            else:
                x, y = self.x_min, self.y_max - (p - 2 * (self.x_max - self.x_min) - (self.y_max - self.y_min))
            x = np.clip(x + np.random.uniform(-randomness, randomness) * (self.x_max - self.x_min) * 0.1,
                        self.x_min, self.x_max)
            y = np.clip(y + np.random.uniform(-randomness, randomness) * (self.y_max - self.y_min) * 0.1,
                        self.y_min, self.y_max)
            waypoints.append(np.array([x, y, self.z]))
        return np.array(waypoints)


def _thrust_to_action(U1, mass, g=9.81):
    hover_T = mass * g
    return float(np.clip((U1 / hover_T - 0.9) / 0.4, -1.0, 1.0))


# ── Main environment class ─────────────────────────────────────────────────────
class UAVPrecompEnv(gymnasium.Env):
    """
    Gymnasium environment wrapping the full EKF + frozen LSTM + PID UAV mission.

    The RL agent outputs per-axis confidence multipliers [α_x, α_y] ∈ [0.0, 2.0]²
    that independently scale the X and Y components of the LSTM drift vector when
    adjusting upcoming waypoints during GPS dropout.  α=1.0 means pure LSTM,
    α=0.0 cancels the correction entirely, >1 amplifies.  Adjusted waypoint
    displacement is hard-capped at MAX_WAYPOINT_SHIFT=0.25 m to prevent extreme
    shifts from making waypoints unreachable.  All other control is handled by
    the existing EKF + PID stack — the RL policy makes one decision per step.
    """

    metadata = {'render_modes': ['human']}
    OBS_DIM  = 42
    ACT_DIM  = 2

    def __init__(self, render_mode=None, lstm_device='cpu'):
        super().__init__()
        self.render_mode  = render_mode
        self.lstm_device  = lstm_device

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.float32(0.0), high=np.float32(2.0),  # V11: widened from 1.5
            shape=(self.ACT_DIM,), dtype=np.float32
        )

        # Load LSTM once and freeze weights
        if not LSTM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"LSTM model not found at {LSTM_MODEL_PATH}.\n"
                "Run collect_pybullet_position_drift_data_precomp.py then "
                "train_lstm_position_drift_precomp.py first."
            )
        self.lstm = PositionDriftLSTMPrecomp(NUM_HORIZONS, 256).to(lstm_device)
        self.lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=lstm_device,
                                             weights_only=True))
        self.lstm.eval()
        for p in self.lstm.parameters():
            p.requires_grad_(False)

        # Physics environment — created once, reset() re-initialises it each episode
        kwargs = {}
        if render_mode == 'human':
            kwargs['render_mode'] = 'human'
        self._penv = DroneMissionEnv(
            physics="PyBulletPhysics",
            control_mode="AttitudeRate",
            drone_model="cf21x_bullet",
            dropout_mode="NONE",
            **kwargs
        )
        self.dt = self._penv.TIME_STEP

        # Controllers (reset inside reset())
        self.quad = QuadcopterPID(dt=self.dt)
        self.kf = QuadcopterEKF(
            dt=self.dt,
            sigma_gyro_bias=EKF_SIGMA_GYRO_BIAS,
            q_bias_floor=EKF_Q_BIAS_FLOOR,
            initial_cov_diag=np.array(
                [
                    0.05, 0.05, 0.05,
                    0.10, 0.10, 0.10,
                    0.05, 0.05, 0.05,
                    0.10, 0.10, 0.10,
                    EKF_INITIAL_GYRO_BIAS_STD,
                    EKF_INITIAL_GYRO_BIAS_STD,
                    EKF_INITIAL_GYRO_BIAS_STD,
                ],
                dtype=float,
            ),
        )
        self.ekf_noise = EKFSensorNoise(
            sample_turn_on_bias_once=True,
            gyro_turn_on_bias_sigma=0.0,
        )

        # Episode state — populated by reset()
        self.waypoints           = None
        self.adjusted_waypoints  = None
        self.lstm_only_waypoints = None   # LSTM-only adjusted (no RL) — used in reward
        self.dropout_windows     = []
        self.mission_time        = 0.0
        self._max_mission_time   = 0.0
        self.current_wp_idx      = 0
        self.reached_waypoints   = 0
        self.path_active         = False
        self.imu_buffer          = np.zeros((BUFFER_LEN, 7), dtype=np.float32)
        self.buffer_idx          = 0
        self.steps_since_dropout = 0
        self.prev_gps_ok         = True
        self._last_lstm_preds    = np.zeros((NUM_HORIZONS, 3), dtype=np.float32)
        self._ekf_pos            = np.zeros(3)
        self._true_pos           = np.zeros(3)
        self._v_true             = np.zeros(3)
        self._gps_ok             = True
        self._t_norm             = 0.0

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs, _ = self._penv.reset()
        # Re-attach attitude controller after physics reset
        self._penv.drone.control = AttitudeRate(
            bc=self._penv.bc,
            drone=self._penv.drone,
            time_step=self._penv.TIME_STEP,
        )
        self._penv.set_target(np.array([0.0, 0.0, TAKEOFF_Z]))

        # Generate a fresh random mission
        gen     = _PatternGenerator(area_size=(3.5, 2.5), altitude=1.0)
        pattern = np.random.choice(['zigzag', 'spiral', 'perimeter'])
        if pattern == 'zigzag':
            self.waypoints = gen.zigzag_search()
        elif pattern == 'spiral':
            self.waypoints = gen.spiral_search()
        else:
            self.waypoints = gen.perimeter_search()

        self.adjusted_waypoints  = self.waypoints.copy()
        self.lstm_only_waypoints = self.waypoints.copy()

        # Mission duration + GPS dropout windows (same logic as precomp demo)
        total_dist = sum(
            np.linalg.norm(self.waypoints[i] - self.waypoints[i - 1])
            for i in range(1, len(self.waypoints))
        )
        T_final = total_dist / TARGET_SPEED + 5.0
        self._max_mission_time = T_final

        num_dropouts = int(T_final / 10)
        self.dropout_windows = []
        for i in range(num_dropouts):
            start = 3.0 + i * (T_final - 4.0) / max(num_dropouts, 1)
            end   = start + 3.5
            if end < T_final - 1.0:
                self.dropout_windows.append((start, end))

        # Reset controllers
        self.quad.reset()
        self.ekf_noise.reset()
        self.kf.reset(
            state=np.concatenate(
                [
                    obs[0:3].copy(),
                    self._penv.drone.xyz_dot.copy(),
                    self._penv.drone.rpy.copy(),
                    self._penv.drone.rpy_dot.copy(),
                    _true_gyro_bias_from_sensor_noise(self.ekf_noise),
                ]
            )
        )

        # Reset episode state
        self.mission_time        = 0.0
        self.current_wp_idx      = 0
        self.reached_waypoints   = 0
        self.path_active         = False
        self.imu_buffer[:]       = 0.0
        self.buffer_idx          = 0
        self.steps_since_dropout = 0
        self.prev_gps_ok         = True
        self._last_lstm_preds[:] = 0.0
        self._true_pos           = self._penv.drone.xyz.copy()
        self._ekf_pos            = obs[0:3].copy()
        self._v_true             = self._penv.drone.xyz_dot.copy()
        self._gps_ok             = True
        self._t_norm             = 0.0
        self._dropout_duration_s = 0.0

        return self._get_obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        self.mission_time += self.dt
        t = self.mission_time

        # GPS availability
        gps_ok       = not any(s <= t < e for s, e in self.dropout_windows)
        self._gps_ok = gps_ok

        # True physics state (ground-truth, always accessible during training)
        true_pos = self._penv.drone.xyz.copy()
        v_true   = self._penv.drone.xyz_dot.copy()
        ang      = self._penv.drone.rpy.copy()
        rate     = self._penv.drone.rpy_dot.copy()
        self._true_pos = true_pos
        self._v_true   = v_true

        # Liftoff detection (barometric altitude — GPS-independent)
        if not self.path_active and true_pos[2] > PATHFOLLOW_TRIGGER_Z:
            self.path_active = True

        # Position source for PID: GPS-on → ground truth; dropout → EKF XY + baro Z
        if gps_ok:
            pos_ctrl = true_pos.copy()
            vel_ctrl = v_true.copy()
        else:
            pos_ctrl = np.array([self._ekf_pos[0], self._ekf_pos[1], true_pos[2]])
            vel_ctrl = v_true.copy()

        # Waypoint arrival check
        waypoint_just_reached = False
        intended_wp_reached   = None
        if self.path_active and self.current_wp_idx < len(self.waypoints):
            dist_to_wp = np.linalg.norm(pos_ctrl - self.adjusted_waypoints[self.current_wp_idx])
            if dist_to_wp < WAYPOINT_TOLERANCE:
                waypoint_just_reached = True
                intended_wp_reached   = self.waypoints[self.current_wp_idx].copy()
                self.reached_waypoints += 1
                self.current_wp_idx    += 1

        target_wp = (self.adjusted_waypoints[self.current_wp_idx]
                     if self.current_wp_idx < len(self.waypoints)
                     else self.adjusted_waypoints[-1])

        # PID control
        self.quad.inject_external_state(pos_ctrl, vel_ctrl, ang, rate)
        z_ref      = self._penv.get_mission_reference()[2] if not self.path_active else target_wp[2]
        ref_target = target_wp[:3] if self.path_active else self._penv.get_mission_reference()
        control_profile = "nominal" if gps_ok else "dropout"
        ctrl = self.quad.step(
            ref_target,
            np.zeros(3),
            z_ref=z_ref,
            freeze_z_integrator=not gps_ok,
            control_profile=control_profile,
        )

        pid_action    = np.zeros(4, dtype=np.float32)
        pid_action[0] = _thrust_to_action(ctrl["thrust_cmd"], self.quad.m, self.quad.g)
        pid_action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        # Step physics
        obs_new, _, done, truncated, _ = self._penv.step(pid_action)

        true_pos = self._penv.drone.xyz.copy()
        v_true = self._penv.drone.xyz_dot.copy()
        ang = self._penv.drone.rpy.copy()
        rate = self._penv.drone.rpy_dot.copy()
        self._true_pos = true_pos
        self._v_true = v_true

        # Tuned 15-state EKF predict/update path.
        noisy_pos, noisy_vel, noisy_att, noisy_rates, _ = self.ekf_noise.add_noise(
            pos=np.asarray(true_pos, dtype=float).reshape(3),
            vel=np.asarray(v_true, dtype=float).reshape(3),
            rot=np.asarray(ang, dtype=float).reshape(3),
            omega=np.asarray(rate, dtype=float).reshape(3),
            acc=np.zeros(3, dtype=float),
            dt=self.dt,
        )
        imu_g = noisy_rates.copy()

        if gps_ok:
            self._dropout_duration_s = 0.0
            self.kf.predict(omega=_motor_omega_from_applied_forces(self._penv, self.kf), dt=self.dt)
        else:
            self._dropout_duration_s += self.dt
            self.kf.predict_dropout(
                omega=_motor_omega_from_applied_forces(self._penv, self.kf),
                dt=self.dt,
                dropout_time=self._dropout_duration_s,
            )

        baro_z = self.ekf_noise.add_noise_to_baro(float(true_pos[2]))
        update_components = [
            self.kf.build_attitude_measurement(noisy_att),
            self.kf.build_gyro_measurement(noisy_rates),
            self.kf.build_baro_measurement(
                baro_z,
                std=self.ekf_noise.baro_noise_std,
            ),
        ]
        if gps_ok:
            update_components.insert(2, self.kf.build_velocity_measurement(noisy_vel))
            update_components.append(self.kf.build_gps_measurement(noisy_pos))
        else:
            update_components.insert(
                2,
                self.kf.build_velocity_pseudo_measurement(
                    np.zeros(3, dtype=float),
                    std_xy=PSEUDO_VEL_STD_XY,
                    std_z=PSEUDO_VEL_STD_Z,
                ),
            )
            odom_velocity = getattr(self._penv, 'odometry_velocity', None)
            if odom_velocity is None:
                odom_velocity = getattr(self._penv.drone, 'odometry_velocity', None)
            odom_position = getattr(self._penv, 'odometry_position', None)
            if odom_position is None:
                odom_position = getattr(self._penv.drone, 'odometry_position', None)
            if odom_position is not None or odom_velocity is not None:
                odom_pos = noisy_pos if odom_position is None else np.asarray(odom_position, dtype=float).reshape(3)
                odom_vel = noisy_vel if odom_velocity is None else np.asarray(odom_velocity, dtype=float).reshape(3)
                update_components.append(
                    self.kf.build_odom_measurement(
                        position=odom_pos,
                        velocity=odom_vel,
                    )
                )
            else:
                flow_velocity_xy = getattr(self._penv, 'optical_flow_velocity_xy', None)
                if flow_velocity_xy is None:
                    flow_velocity_xy = getattr(self._penv.drone, 'optical_flow_velocity_xy', None)
                if flow_velocity_xy is not None:
                    update_components.append(
                        self.kf.build_optical_flow_measurement(flow_velocity_xy)
                    )

        for z_u, H_u, R_u in update_components:
            self.kf.update(measurement=z_u, H=H_u, measurement_noise=R_u)

        self._ekf_pos = self.kf.position.copy()

        # t_norm and dropout counter
        if self.prev_gps_ok and not gps_ok:
            self.steps_since_dropout = 0                 # GPS just dropped
        elif not self.prev_gps_ok and gps_ok:
            self.steps_since_dropout = 0                 # GPS just recovered
            self.adjusted_waypoints  = self.waypoints.copy()
            self.lstm_only_waypoints = self.waypoints.copy()
        elif not gps_ok:
            self.steps_since_dropout += 1
        self._t_norm = min(self.steps_since_dropout / SHADOW_RESET_INTERVAL, 1.0)

        # IMU buffer update
        slot = self.buffer_idx % BUFFER_LEN
        self.imu_buffer[slot] = np.concatenate([v_true, imu_g, [self._t_norm]])
        self.buffer_idx += 1

        # LSTM inference + apply LSTM prediction + RL residual action (dropout only)
        if self.buffer_idx >= BUFFER_LEN and not gps_ok:
            roll_by     = -(self.buffer_idx % BUFFER_LEN)
            imu_ordered = np.roll(self.imu_buffer, roll_by, axis=0)
            tensor      = torch.from_numpy(imu_ordered).unsqueeze(0).to(self.lstm_device)
            with torch.no_grad():
                preds = self.lstm(tensor)            # (1, 7, 3)
            self._last_lstm_preds = preds[0].cpu().numpy()

            alpha_x = float(action[0])   # per-axis confidence multipliers ∈ [0.0, 1.5]
            alpha_y = float(action[1])
            for wi in range(self.current_wp_idx,
                            min(self.current_wp_idx + 5, len(self.waypoints))):
                dist    = float(np.linalg.norm(self._ekf_pos - self.waypoints[wi]))
                t_arr_s = dist / TARGET_SPEED if dist > 1e-3 else 0.0
                h_idx   = min(int(t_arr_s / (HORIZON_STRIDE * self.dt)), NUM_HORIZONS - 1)
                drift   = self._last_lstm_preds[h_idx].copy()
                drift[2] = 0.0   # XY only — barometer is GPS-independent
                # Project correction onto approach direction so cross-track component
                # doesn't push turn waypoints past boundaries (zigzag overshoot fix)
                approach_xy = self.waypoints[wi][:2] - self._ekf_pos[:2]
                a_norm      = np.linalg.norm(approach_xy)
                if a_norm > 1e-3:
                    u           = approach_xy / a_norm
                    drift_along = np.dot(drift[:2], u) * u          # parallel only
                else:
                    drift_along = drift[:2]
                self.lstm_only_waypoints[wi] = self.waypoints[wi] + np.array([drift_along[0], drift_along[1], 0.0])
                shift_xy  = np.array([alpha_x * drift_along[0], alpha_y * drift_along[1]])
                shift_mag = np.linalg.norm(shift_xy)
                if shift_mag > MAX_WAYPOINT_SHIFT:        # V11: cap displacement
                    shift_xy = shift_xy * (MAX_WAYPOINT_SHIFT / shift_mag)
                self.adjusted_waypoints[wi]  = self.waypoints[wi] + np.array([shift_xy[0], shift_xy[1], 0.0])
        elif gps_ok:
            self._last_lstm_preds[:] = 0.0

        self.prev_gps_ok = gps_ok

        # ── Reward (V8 — true counterfactual) ─────────────────────────────────
        reward = 0.0

        # Dense counterfactual: how much does RL's α reduce drift error vs LSTM-only?
        # ekf_drift is the actual accumulated XY error — ground truth available in training.
        # Optimal α per axis: ekf_drift[i] / dp[i]  (policy learns this ratio directly).
        if not gps_ok and self.current_wp_idx < len(self.waypoints) and self.buffer_idx >= BUFFER_LEN:
            ekf_drift = self._ekf_pos[:2] - true_pos[:2]          # actual XY EKF drift
            wi      = self.current_wp_idx
            dist    = float(np.linalg.norm(self._ekf_pos - self.waypoints[wi]))
            t_arr_s = dist / TARGET_SPEED if dist > 1e-3 else 0.0
            h_idx   = min(int(t_arr_s / (HORIZON_STRIDE * self.dt)), NUM_HORIZONS - 1)
            dp      = self._last_lstm_preds[h_idx][:2]             # LSTM XY drift prediction
            alpha_x = float(action[0])
            alpha_y = float(action[1])
            # Expected arrival error: |correction − actual drift|
            lstm_err = float(np.sqrt((dp[0] - ekf_drift[0])**2 + (dp[1] - ekf_drift[1])**2))
            rl_err   = float(np.sqrt((alpha_x * dp[0] - ekf_drift[0])**2 +
                                     (alpha_y * dp[1] - ekf_drift[1])**2))
            reward  += (lstm_err - rl_err) * 2.0                   # positive when RL beats LSTM

        # Arrival reward: accuracy at the moment of reaching a waypoint
        if waypoint_just_reached and intended_wp_reached is not None:
            arrival_err = np.linalg.norm(true_pos[:2] - intended_wp_reached[:2])
            reward += max(0.0, 1.0 - arrival_err / 0.5)   # 1.0 perfect → 0.0 at ≥50 cm

        # Tiny regularisation — tie-breaker only, does not dominate signal
        reward -= 0.001 * ((float(action[0]) - 1.0)**2 + (float(action[1]) - 1.0)**2)

        # ── Termination ───────────────────────────────────────────────────────
        all_reached = self.current_wp_idx >= len(self.waypoints)
        timeout     = t >= self._max_mission_time

        info_out = {
            'mission_time':      t,
            'gps_ok':            gps_ok,
            'true_pos':          true_pos,
            'ekf_pos':           self._ekf_pos.copy(),
            'ekf_error':         float(np.linalg.norm(self._ekf_pos - true_pos)),
            'waypoints_reached': self.reached_waypoints,
            'total_waypoints':   len(self.waypoints),
            't_norm':            self._t_norm,
            'dropout_duration_s': float(self._dropout_duration_s),
            'control_profile':    control_profile,
            'rl_action':         action.copy(),
        }

        terminated = bool(all_reached or done)
        truncated  = bool(truncated or timeout)
        return self._get_obs(), float(reward), terminated, truncated, info_out

    # ── Observation builder ───────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        # [0:21] LSTM 7-horizon predictions, flattened, clipped ±1 m
        lstm_flat = np.clip(self._last_lstm_preds.flatten(), -1.0, 1.0).astype(np.float32)

        # [21] t_norm
        t_norm = np.float32(self._t_norm)

        # [22:25] velocity clipped ±5 m/s
        vel = np.clip(self._v_true, -5.0, 5.0).astype(np.float32)

        # [25:27] normalised XY direction to current waypoint
        if self.waypoints is not None and self.current_wp_idx < len(self.waypoints):
            diff       = self.waypoints[self.current_wp_idx][:2] - self._ekf_pos[:2]
            dist       = float(np.linalg.norm(diff))
            wp_dir     = (diff / dist if dist > 1e-3 else np.zeros(2)).astype(np.float32)
            dist_to_wp = np.float32(min(dist, 5.0))
        else:
            wp_dir     = np.zeros(2, dtype=np.float32)
            dist_to_wp = np.float32(0.0)

        # [28] GPS status flag
        gps_flag = np.float32(1.0 if self._gps_ok else 0.0)

        # [29:32] EKF XYZ position standard deviation — grows during dropout
        ekf_pos_std = np.sqrt(np.clip(np.diag(self.kf.P)[:3], 0.0, None))
        ekf_pos_std = np.clip(ekf_pos_std, 0.0, 5.0).astype(np.float32)

        # [32:34] applied alpha per axis (centred: alpha-1), so 0.0 = pure LSTM
        if (self.adjusted_waypoints is not None
                and self.lstm_only_waypoints is not None
                and self.current_wp_idx < len(self.waypoints)):
            wi = self.current_wp_idx
            lstm_drift_xy = (self.lstm_only_waypoints[wi][:2]
                             - self.waypoints[wi][:2])
            # α × drift − drift = (α−1) × drift; clamp to ±1.0 for normalisation
            # (expanded from ±0.5 in V10 to match wider action space [0.0, 1.5])
            applied_alpha_xy = np.clip(
                self.adjusted_waypoints[wi][:2] - self.lstm_only_waypoints[wi][:2],
                -1.0, 1.0
            ).astype(np.float32)
        else:
            applied_alpha_xy = np.zeros(2, dtype=np.float32)

        # [34:40] Lookahead segment direction vectors (3 × XY-normalised)
        # v0: wp[i]→wp[i+1], v1: wp[i+1]→wp[i+2], v2: wp[i+2]→wp[i+3]
        # [40:42] Segment lengths: [dist(wp[i]→wp[i+1]), dist(wp[i+1]→wp[i+2])], clipped [0, 5]
        lookahead_vecs = np.zeros(6, dtype=np.float32)
        lookahead_lens = np.zeros(2, dtype=np.float32)
        if self.waypoints is not None:
            n_wps = len(self.waypoints)
            wi    = self.current_wp_idx
            for j in range(3):
                a, b = wi + j, wi + j + 1
                if b < n_wps:
                    seg     = self.waypoints[b][:2] - self.waypoints[a][:2]
                    seg_len = float(np.linalg.norm(seg))
                    lookahead_vecs[j*2:j*2+2] = (
                        (seg / seg_len).astype(np.float32) if seg_len > 1e-3
                        else np.zeros(2, dtype=np.float32)
                    )
                    if j < 2:
                        lookahead_lens[j] = np.float32(min(seg_len, 5.0))

        return np.concatenate([lstm_flat, [t_norm], vel, wp_dir, [dist_to_wp],
                                [gps_flag], ekf_pos_std, applied_alpha_xy,
                                lookahead_vecs, lookahead_lens])

    def close(self):
        try:
            self._penv.close()
        except Exception:
            pass
