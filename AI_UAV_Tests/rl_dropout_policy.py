"""Residual PPO environment for dropout-aware trajectory tracking.

Control stack:
    trajectory -> EKF -> PID baseline -> residual PPO -> mixer -> drone

The policy always acts, but only as a bounded residual on top of the PID
baseline. The observation exposes the EKF state, uncertainty, dropout flag,
reference trajectory, and the exact baseline control preview `u_pid`.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.followpath_dropout_mission import (
    DroneFollowPathDropoutMissionEnv,
)
from phoenix_drone_simulation.envs.sensors import SensorNoise
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter
from AI_UAV_Tests.trajectories_library import FlightMission


def build_dropout_training_mission(task: str) -> FlightMission:
    task_name = str(task).strip().lower()
    mission = FlightMission(default_z=1.0, ground_z=0.0)

    if task_name == "hover":
        mission.add_takeoff(duration=3.0, target_z=1.0)
        mission.add_hover(duration=16.0, z=1.0, pos=(0.0, 0.0, 1.0))
        mission.add_landing(duration=4.0, ground_z=0.055)
        return mission

    if task_name == "circle":
        mission.add_takeoff(duration=3.0, target_z=1.0)
        mission.add_circle(
            duration=16.0,
            radius=0.5,
            period=12.0,
            z=1.0,
            center_xy=(0.0, 0.0),
        )
        mission.add_hover(duration=2.0, z=1.0)
        mission.add_landing(duration=4.0, ground_z=0.055)
        return mission

    if task_name == "full":
        mission.add_takeoff(duration=3.0, target_z=1.0)
        mission.add_circle(duration=12.0, radius=0.5, period=12.0, z=1.0, center_xy=(0.0, 0.0))
        mission.add_hover(duration=2.0, z=1.0)
        mission.add_landing(duration=6.0, ground_z=0.055)
        return mission

    raise ValueError(f"Unknown task '{task}'. Expected one of: hover, circle, full.")


class DroneDropoutRLEnv(gym.Env):
    """Gym environment for residual PPO on top of EKF + PID."""

    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        mission: FlightMission | None = None,
        task: str = "full",
        render_mode: str | None = None,
        dropout_randomize: bool = True,
        max_episode_steps: int | None = None,
        residual_alpha: float = 0.25,
        track_trajectory_during_dropout: bool = True,
        adaptive_residual_scaling: bool = True,
        altitude_ceiling_m: float = 2.5,
        residual_thrust_limit_scale: float = 0.35,
        residual_tau_limit_scale: float = 0.20,
    ):
        self.task = str(task).strip().lower()
        self.render_mode = render_mode
        self.dropout_randomize = dropout_randomize
        self.max_episode_steps = max_episode_steps
        self.alpha = float(residual_alpha)
        self.track_trajectory_during_dropout = bool(track_trajectory_during_dropout)
        self.adaptive_residual_scaling = bool(adaptive_residual_scaling)
        self.altitude_ceiling_m = float(altitude_ceiling_m)
        self.residual_thrust_limit_scale = float(residual_thrust_limit_scale)
        self.residual_tau_limit_scale = float(residual_tau_limit_scale)

        # Build mission
        if mission is None:
            mission = build_dropout_training_mission(self.task)
        self.mission = mission

        # Create environment
        self.env = DroneFollowPathDropoutMissionEnv(
            trajectory_fn=self.mission,
            physics="PyBulletPhysics",
            control_mode="AttitudeRate",
            drone_model="cf21x_bullet",
            dropout_mode="NONE",
            render_mode="human" if render_mode == "human" else None,
            observation_noise=1.0,
        )
        self.env.drone.control = AttitudeRate(
            bc=self.env.bc,
            drone=self.env.drone,
            time_step=self.env.TIME_STEP,
        )
        self.env.reset()
        if self.max_episode_steps is None:
            mission_horizon_s = max(float(getattr(self.mission, "total_time", 0.0)), 10.0)
            self.max_episode_steps = max(
                1, int(np.ceil(mission_horizon_s / float(self.env.TIME_STEP)))
            )
        else:
            self.max_episode_steps = int(self.max_episode_steps)

        # Controllers
        self.quad = QuadcopterPID(dt=self.env.TIME_STEP)
        self.quad.reset()

        self.ekf = PhoenixEKFAdapter(dt=self.env.TIME_STEP)
        self.ekf.reset(
            position=self.env.drone.xyz,
            velocity=self.env.drone.xyz_dot,
            attitude=self.env.drone.rpy,
            rates=self.env.drone.rpy_dot,
        )

        self.noise_gen = SensorNoise()
        self.last_ctrl_pos = self.env.drone.xyz.copy()

        # Dropout parameters (randomized per episode)
        self.dropout_start = 5.0
        self.dropout_duration = 4.0
        self.dropout_triggered = False
        self.dropout_done = False

        # State/action spaces
        # State: EKF state + uncertainty + dropout context + ref errors + u_pid.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32
        )

        # Action: residuals on body-level commands: [dU1, dTau_phi, dTau_theta, dTau_psi]
        # normalized to [-1, 1]. The environment will map these to physical deltas.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Tracking
        self.step_count = 0
        self.episode_return = 0.0
        self.last_pos_error = 0.0
        self.last_valid_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        self._dropout_was_active = False
        self.prev_pos_err = 0.0
        self.prev_vel_err = 0.0
        self.dropout_error_sum = 0.0
        self.dropout_step_count = 0
        self._dropout_z_int_hold = 0.0
        self._dropout_thrust_hold = 0.0

    def _randomize_dropout_schedule(self) -> None:
        """Randomize dropout start time and duration for domain randomization."""
        episode_horizon_s = float(self.max_episode_steps) * float(self.env.TIME_STEP)
        earliest_start = min(4.0, 0.4 * episode_horizon_s)
        latest_start = max(earliest_start + 0.5, 0.7 * episode_horizon_s)
        latest_start = min(latest_start, episode_horizon_s - 1.0)

        if self.dropout_randomize:
            self.dropout_start = float(np.random.uniform(earliest_start, latest_start))
            max_duration = max(1.0, min(6.0, episode_horizon_s - self.dropout_start - 0.25))
            self.dropout_duration = float(np.random.uniform(1.0, max_duration))
        else:
            self.dropout_start = float(min(5.0, max(earliest_start, 0.45 * episode_horizon_s)))
            self.dropout_duration = float(min(4.0, max(1.5, episode_horizon_s - self.dropout_start - 0.25)))

    def _dropout_context(self) -> tuple[float, float, float]:
        dropout_time = float(getattr(self.ekf, "dropout_time", 0.0))
        dropout_ratio = np.clip(dropout_time / max(self.dropout_duration, self.env.TIME_STEP), 0.0, 1.0)
        remaining_ratio = np.clip(
            max(self.dropout_duration - dropout_time, 0.0) / max(self.dropout_duration, self.env.TIME_STEP),
            0.0,
            1.0,
        )
        return dropout_time, float(dropout_ratio), float(remaining_ratio)

    def _effective_alpha(self, std_pos: np.ndarray) -> float:
        if not self.adaptive_residual_scaling:
            return self.alpha

        dropout_flag = 1.0 if self.env.dropout_mgr.active else 0.0
        _, dropout_ratio, _ = self._dropout_context()
        uncertainty_ratio = np.clip(float(np.linalg.norm(std_pos)) / 0.20, 0.0, 1.0)
        scale = 0.35 + 0.65 * max(dropout_flag * dropout_ratio, uncertainty_ratio * dropout_flag)
        return float(self.alpha * scale)

    def _controller_snapshot(self) -> dict:
        return {
            "x": self.quad.x.copy(),
            "v": self.quad.v.copy(),
            "ang": self.quad.ang.copy(),
            "rate": self.quad.rate.copy(),
            "z_int": float(self.quad.z_int),
            "att_int": self.quad.att_int.copy(),
            "rate_int": self.quad.rate_int.copy(),
            "use_external_state": bool(self.quad.use_external_state),
        }

    def _restore_controller_snapshot(self, snapshot: dict) -> None:
        self.quad.x = snapshot["x"].copy()
        self.quad.v = snapshot["v"].copy()
        self.quad.ang = snapshot["ang"].copy()
        self.quad.rate = snapshot["rate"].copy()
        self.quad.z_int = float(snapshot["z_int"])
        self.quad.att_int = snapshot["att_int"].copy()
        self.quad.rate_int = snapshot["rate_int"].copy()
        self.quad.use_external_state = bool(snapshot["use_external_state"])

    def _reference_for_policy(self) -> tuple[np.ndarray, np.ndarray]:
        pos_ref, vel_ref = self.env.current_reference()
        pos_ref = np.asarray(pos_ref, dtype=float)
        vel_ref = np.asarray(vel_ref, dtype=float)

        if self.env.dropout_mgr.active and not self.track_trajectory_during_dropout:
            return self.last_valid_ref.copy(), np.zeros(3, dtype=float)

        self.last_valid_ref = pos_ref.copy()
        return pos_ref, vel_ref

    def _inject_estimate_into_pid(self) -> dict:
        estimate = self.ekf.ekf.as_dict()
        self.quad.inject_external_state(
            estimate["x"], estimate["v"], estimate["ang"], estimate["rate"]
        )
        return estimate

    def _compute_pid_output(
        self,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
        *,
        freeze_z_integrator: bool = False,
    ) -> dict:
        self._inject_estimate_into_pid()
        return self.quad.step(
            ref_pos,
            ref_vel,
            z_ref=float(ref_pos[2]),
            freeze_z_integrator=freeze_z_integrator,
        )

    def _preview_pid_output(
        self,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
        *,
        freeze_z_integrator: bool = False,
    ) -> dict:
        snapshot = self._controller_snapshot()
        try:
            return self._compute_pid_output(
                ref_pos,
                ref_vel,
                freeze_z_integrator=freeze_z_integrator,
            )
        finally:
            self._restore_controller_snapshot(snapshot)

    def _build_state(self, pid_ctrl: dict, ref_pos: np.ndarray, ref_vel: np.ndarray) -> np.ndarray:
        """Build the policy observation."""
        ekf_state = self.ekf.ekf.as_dict()
        pos = ekf_state["x"]
        vel = ekf_state["v"]
        att = ekf_state["ang"]
        rate = ekf_state["rate"]

        # Covariance uncertainty (standard deviations)
        P_diag = np.diag(self.ekf.ekf.P)
        std = np.sqrt(np.maximum(P_diag, 1e-9))
        std_pos = std[0:3]
        std_vel = std[3:6]
        std_att = std[6:9]
        std_rate = std[9:12]

        dropout_flag = float(self.env.dropout_mgr.active)
        dropout_time, dropout_ratio, remaining_ratio = self._dropout_context()
        alpha_effective = self._effective_alpha(std_pos)
        pos_err_vec = ref_pos - pos
        vel_err_vec = ref_vel - vel

        state = np.concatenate([
            pos, vel, att, rate,
            std_pos, std_vel, std_att, std_rate,
            [dropout_flag, dropout_time, dropout_ratio, remaining_ratio],
            ref_pos, ref_vel,
            pos_err_vec, vel_err_vec,
        ])

        # Append baseline controller output (u_pid) normalized.
        U1_pid = float(pid_ctrl["thrust_cmd"])
        tau_pid = np.asarray(pid_ctrl.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)

        # Normalize: thrust via same mapping as _thrust_to_action, taus by tau_max
        thrust_norm = self._thrust_to_action(U1_pid, self.quad.m)
        tau_norm = np.clip(tau_pid / float(self.quad.tau_max), -1.0, 1.0)

        state = np.concatenate([state, [alpha_effective, thrust_norm], tau_norm]).astype(np.float32)

        return state

    def _apply_dropout_baseline_safeguards(
        self,
        pid_ctrl: dict,
        *,
        is_drop: bool,
        just_started_dropout: bool,
    ) -> dict:
        pid_safe = dict(pid_ctrl)
        if just_started_dropout:
            self._dropout_thrust_hold = float(pid_ctrl["thrust_cmd"])

        if is_drop:
            thrust_hold = float(np.clip(
                self._dropout_thrust_hold,
                0.5 * self.quad.m * self.quad.g,
                1.1 * self.quad.m * self.quad.g,
            ))
            pid_safe["thrust_cmd"] = thrust_hold

        return pid_safe

    def _action_to_control(self, action: np.ndarray, pid_ctrl: dict) -> dict:
        """Convert residual RL action to control commands.

        Action: normalized residuals in [-1,1] representing physical deltas:
            delta_u_phys = action * delta_limits
        Then: u_final = u_pid + alpha * delta_u_phys
        """
        # Ensure correct shape
        action = np.asarray(action, dtype=float).reshape(4)
        U1_pid = float(pid_ctrl["thrust_cmd"])
        tau_pid = np.asarray(pid_ctrl.get("tau_cmd", np.zeros(3)), dtype=float).reshape(3)
        rates_des_pid = np.asarray(pid_ctrl.get("rates_des", np.zeros(3)), dtype=float).reshape(3)
        std_pos = np.sqrt(np.maximum(np.diag(self.ekf.ekf.P)[0:3], 1e-9))
        alpha_eff = self._effective_alpha(std_pos)

        # Compute physical delta limits
        delta_limits = np.array([
            self.residual_thrust_limit_scale * self.quad.m * self.quad.g,
            self.residual_tau_limit_scale * self.quad.tau_max,
            self.residual_tau_limit_scale * self.quad.tau_max,
            self.residual_tau_limit_scale * self.quad.tau_max,
        ], dtype=float)

        # Map normalized action to physical residuals and clip
        delta_u = (action * delta_limits).astype(float)
        delta_u = np.clip(delta_u, -delta_limits, delta_limits)

        # Form final body-level commands
        U1_final = float(np.clip(
            U1_pid + alpha_eff * delta_u[0],
            0.5 * self.quad.m * self.quad.g,
            1.3 * self.quad.m * self.quad.g,
        ))
        tau_final = np.clip(tau_pid + alpha_eff * delta_u[1:4], -self.quad.tau_max, self.quad.tau_max)

        # Approximate desired rates that would produce tau_final by inverting P gain
        tau_delta_applied = tau_final - tau_pid
        rates_des_final = rates_des_pid + tau_delta_applied / np.maximum(self.quad.Kp_rate, 1e-8)
        rates_des_final = np.clip(rates_des_final, -self.quad.max_rate, self.quad.max_rate)

        omega = self.quad.mixer(U1_final, tau_final)

        return {
            "u_pid": np.concatenate([[U1_pid], tau_pid]),
            "u_final": np.concatenate([[U1_final], tau_final]),
            "thrust_cmd": U1_final,
            "rates_des": rates_des_final,
            "omega_cmd": omega,
            "tau_cmd": tau_final,
            "delta_u_phys": delta_u,
            "alpha_effective": alpha_eff,
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """One environment step."""
        self.step_count += 1

        # Apply dropout schedule
        t = float(self.env.mission_time)
        if not self.dropout_triggered and t >= self.dropout_start:
            self.env.dropout_mgr.mode = "HOV"
            self.env.trigger_dropout()
            self.dropout_triggered = True

        if self.dropout_triggered and not self.dropout_done and t >= self.dropout_start + self.dropout_duration:
            self.env.clear_dropout()
            self.dropout_done = True

        is_drop = bool(self.env.dropout_mgr.active)
        just_started_dropout = is_drop and not self._dropout_was_active
        just_recovered = self._dropout_was_active and not is_drop

        if just_started_dropout:
            self._dropout_z_int_hold = float(self.quad.z_int)
        if is_drop:
            self.quad.z_int = float(self._dropout_z_int_hold)

        pos_ref, vel_ref = self._reference_for_policy()

        # Controller advances exactly once per environment step.
        pid_ctrl = self._compute_pid_output(
            pos_ref,
            vel_ref,
            freeze_z_integrator=is_drop,
        )
        pid_ctrl = self._apply_dropout_baseline_safeguards(
            pid_ctrl,
            is_drop=is_drop,
            just_started_dropout=just_started_dropout,
        )
        residual_ctrl = self._action_to_control(action, pid_ctrl)
        ctrl = {
            "thrust_cmd": residual_ctrl["thrust_cmd"],
            "rates_des": residual_ctrl["rates_des"],
            "omega_cmd": residual_ctrl["omega_cmd"],
            "tau_cmd": residual_ctrl.get("tau_cmd", pid_ctrl.get("tau_cmd")),
        }
        self.last_ctrl_pos = self.ekf.ekf.as_dict()["x"].copy()

        # Physics step
        action_env = np.zeros(4, dtype=np.float32)
        action_env[0] = self._thrust_to_action(ctrl["thrust_cmd"], self.quad.m)
        action_env[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        _, _, done_env, _, info = self.env.step(action_env)

        # EKF update
        self.ekf.step(
            motor_omega=ctrl["omega_cmd"],
            position=self.env.drone.xyz,
            velocity=self.env.drone.xyz_dot,
            attitude=self.env.drone.rpy,
            rates=self.env.drone.rpy_dot,
            dropout_active=is_drop,
            dt=self.env.TIME_STEP,
        )

        # Reward uses the current mission trajectory, not a dropout-specific mode switch.
        reward_ref_pos, reward_ref_vel = self._reference_for_policy()
        pos_err = np.linalg.norm(self.env.drone.xyz - reward_ref_pos)
        vel_err = np.linalg.norm(self.env.drone.xyz_dot - reward_ref_vel)
        uncertainty = float(np.trace(self.ekf.ekf.P))
        std_pos = np.sqrt(np.maximum(np.diag(self.ekf.ekf.P)[0:3], 1e-9))
        uncertainty_pos = float(np.linalg.norm(std_pos))
        delta_u_phys = residual_ctrl.get("delta_u_phys", np.zeros(4, dtype=float))
        control_effort = float(np.linalg.norm(np.asarray(action, dtype=float)))
        tilt = float(np.linalg.norm(self.env.drone.rpy[:2]))
        rates_mag = float(np.linalg.norm(self.env.drone.rpy_dot))
        progress_reward = 6.0 * (self.prev_pos_err - pos_err) + 1.5 * (self.prev_vel_err - vel_err)
        in_tolerance = pos_err < 0.08 and vel_err < 0.15 and tilt < np.deg2rad(10.0)

        reward = (
            -6.0 * pos_err
            -0.8 * vel_err
            -0.5 * tilt
            -0.02 * rates_mag
            -0.015 * control_effort
            -0.15 * uncertainty_pos
            + progress_reward
            + 0.15
        )

        if is_drop:
            reward += 0.35
            reward -= 8.0 * pos_err
            reward -= 0.25 * uncertainty_pos
            if pos_err < 0.20:
                reward += 0.25
            if pos_err > 0.60:
                reward -= 8.0 * (pos_err - 0.60)
            self.dropout_error_sum += pos_err
            self.dropout_step_count += 1

        if in_tolerance:
            reward += 0.5

        if just_recovered:
            reward += 6.0 * np.exp(-4.0 * pos_err)

        altitude_runaway = self.env.drone.xyz[2] > self.altitude_ceiling_m
        crashed = self.env.drone.xyz[2] < 0.05 or altitude_runaway
        if crashed:
            reward -= 50.0
        if altitude_runaway:
            reward -= 75.0

        self.last_pos_error = pos_err if is_drop else 0.0
        self.prev_pos_err = pos_err
        self.prev_vel_err = vel_err
        self.episode_return += reward

        truncated = self.step_count >= self.max_episode_steps
        done = bool(done_env or crashed)
        avg_dropout_err = (
            self.dropout_error_sum / self.dropout_step_count if self.dropout_step_count > 0 else np.inf
        )
        success = bool((done or truncated) and not crashed and avg_dropout_err < 0.20)
        if success:
            reward += 25.0
            self.episode_return += 25.0

        next_ref_pos, next_ref_vel = self._reference_for_policy()
        next_pid_preview = self._preview_pid_output(
            next_ref_pos,
            next_ref_vel,
            freeze_z_integrator=bool(self.env.dropout_mgr.active),
        )
        next_pid_preview = self._apply_dropout_baseline_safeguards(
            next_pid_preview,
            is_drop=bool(self.env.dropout_mgr.active),
            just_started_dropout=False,
        )
        state = self._build_state(next_pid_preview, next_ref_pos, next_ref_vel)
        self._dropout_was_active = bool(self.env.dropout_mgr.active)
        info["episode_return"] = float(self.episode_return)
        info["crashed"] = crashed
        info["dropout_active"] = bool(self.env.dropout_mgr.active)
        info["pos_error"] = float(pos_err)
        info["drone_z"] = float(self.env.drone.xyz[2])
        info["altitude_runaway"] = bool(altitude_runaway)
        info["altitude_ceiling_m"] = float(self.altitude_ceiling_m)
        info["cov_trace"] = uncertainty
        info["uncertainty"] = uncertainty
        info["vel_error"] = float(vel_err)
        info["u_pid"] = residual_ctrl["u_pid"].astype(np.float32)
        info["u_final"] = residual_ctrl["u_final"].astype(np.float32)
        info["delta_u_phys"] = delta_u_phys.astype(np.float32)
        info["alpha_effective"] = float(residual_ctrl["alpha_effective"])
        info["pid_z_int"] = float(self.quad.z_int)
        info["dropout_z_int_hold"] = float(self._dropout_z_int_hold)
        info["dropout_thrust_hold"] = float(self._dropout_thrust_hold)
        info["dropout_time"] = float(getattr(self.ekf, "dropout_time", 0.0))
        info["dropout_duration"] = float(self.dropout_duration)
        info["avg_dropout_error"] = float(avg_dropout_err if np.isfinite(avg_dropout_err) else -1.0)
        info["success"] = success
        info["reward"] = float(reward)

        # DEBUG: Alert if episode ends at step 1
        if done and self.step_count == 1:
            print(f"\n[ALERT] Episode terminated at step 1!")
            print(f"  step_count={self.step_count}")
            print(f"  crashed={crashed}")
            print(f"  done (from env)={done or truncated}")
            print(f"  truncated={truncated}")
            print(f"  drone.xyz={self.env.drone.xyz}")
            print(f"  drone.z={self.env.drone.xyz[2]}")
            print(f"  reward={reward}")
            print(f"  dropout_triggered={self.dropout_triggered}")
            print(f"  mission_time={float(self.env.mission_time)}")
            print(f"  info={info}\n")

        return state, float(reward), done, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset environment, handling PyBullet state restoration failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.env.reset()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  [Retry {attempt+1}/{max_retries}] Reset failed: {type(e).__name__}")
                    # Try again on next iteration
                    continue
                else:
                    # Last retry failed, recreate environment from scratch
                    print(f"  [Reset] Recreating environment after {max_retries} failed resets...")
                    try:
                        self.env.close()
                    except:
                        pass

                    self.env = DroneFollowPathDropoutMissionEnv(
                        trajectory_fn=self.mission,
                        physics="PyBulletPhysics",
                        control_mode="AttitudeRate",
                        drone_model="cf21x_bullet",
                        dropout_mode="NONE",
                        render_mode="human" if self.render_mode == "human" else None,
                        observation_noise=1.0,
                    )
                    self.env.drone.control = AttitudeRate(
                        bc=self.env.bc,
                        drone=self.env.drone,
                        time_step=self.env.TIME_STEP,
                    )
                    self.env.reset()

        # Force drone to spawn at safe takeoff altitude (PyBullet spawn bug workaround)
        import pybullet as p
        if self.env.drone.xyz[2] < 0.5:
            p.resetBasePositionAndOrientation(
                self.env.drone.body_unique_id,
                [self.env.drone.xyz[0], self.env.drone.xyz[1], 1.0],
                self.env.drone.quaternion
            )
            # Let physics settle
            for _ in range(50):
                self.env.step(np.zeros(4, dtype=np.float32))
            self.env.mission_time = 0.0
            pos_ref, _ = self.env.current_reference(0.0)
            self.env.set_target(pos_ref)

        self.quad.reset()
        self.ekf.reset(
            position=self.env.drone.xyz,
            velocity=self.env.drone.xyz_dot,
            attitude=self.env.drone.rpy,
            rates=self.env.drone.rpy_dot,
        )

        self.step_count = 0
        self.episode_return = 0.0
        self.dropout_triggered = False
        self.dropout_done = False
        self.last_ctrl_pos = self.env.drone.xyz.copy()
        self.last_valid_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        self._dropout_was_active = False
        self.dropout_error_sum = 0.0
        self.dropout_step_count = 0

        self._randomize_dropout_schedule()

        ref_pos, ref_vel = self._reference_for_policy()
        self.prev_pos_err = float(np.linalg.norm(self.env.drone.xyz - ref_pos))
        self.prev_vel_err = float(np.linalg.norm(self.env.drone.xyz_dot - ref_vel))
        pid_preview = self._preview_pid_output(ref_pos, ref_vel)
        state = self._build_state(pid_preview, ref_pos, ref_vel)
        self._dropout_z_int_hold = float(self.quad.z_int)
        self._dropout_thrust_hold = float(pid_preview["thrust_cmd"])
        return state, {}

    def render(self):
        """Render the environment."""
        pass

    def close(self):
        """Close the environment."""
        if hasattr(self, "env"):
            self.env.close()

    @staticmethod
    def _thrust_to_action(U1: float, mass: float, g: float = 9.81) -> float:
        """Convert thrust command to normalized action."""
        hover_T = mass * g
        a0 = (U1 / hover_T - 0.9) / 0.4
        return float(np.clip(a0, -1.0, 1.0))
