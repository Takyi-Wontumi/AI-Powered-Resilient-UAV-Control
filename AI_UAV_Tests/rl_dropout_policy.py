"""
RL Policy for Dropout Handling
===============================
The RL agent learns to fly during GPS dropout by observing:
- EKF state estimates (position, velocity, attitude, rates)
- EKF uncertainty (covariance diagonal)
- Dropout status flag

During normal flight, the PID controller is used.
During dropout, the RL policy takes over.

This is approach C: RL handles dropout phase only.
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


class DroneDropoutRLEnv(gym.Env):
    """Gym environment for RL training on dropout handling.

    The agent receives the full EKF state (position, velocity, attitude, rates)
    plus the EKF covariance diagonal (uncertainty in each state dimension) plus
    a dropout flag. During dropout, the agent controls attitude+thrust commands.
    During normal flight, PID is used (agent is idle).

    Reward:
    - Negative tracking error (minimize distance from reference)
    - Penalty for high uncertainty (encourage using low-uncertainty estimates)
    - Recovery bonus when exiting dropout
    - Crash penalty
    - Control effort regularization
    """

    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        mission: FlightMission | None = None,
        render_mode: str | None = None,
        dropout_randomize: bool = True,
        max_episode_steps: int = 2000,
    ):
        self.render_mode = render_mode
        self.dropout_randomize = dropout_randomize
        self.max_episode_steps = max_episode_steps

        # Build mission
        if mission is None:
            mission = FlightMission(default_z=1.0, ground_z=0.0)
            mission.add_takeoff(duration=3.0, target_z=1.0)
            mission.add_circle(duration=12.0, radius=0.5, period=12.0, z=1.0, center_xy=(0.0, 0.0))
            mission.add_hover(duration=2.0, z=1.0)
            mission.add_landing(duration=6.0, ground_z=0.055)

        # Create environment
        self.env = DroneFollowPathDropoutMissionEnv(
            trajectory_fn=mission,
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
        # State: [pos(3), vel(3), att(3), rate(3), std_pos(3), std_vel(3), std_att(3), std_rate(3),
        #         dropout_flag(1), ref_pos(3), ref_vel(3)] = 31 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        # Action: [thrust_cmd, roll_des, pitch_des, yaw_rate_des] normalized to [-1, 1]
        # These are only used during dropout; during normal flight, PID is used
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Tracking
        self.step_count = 0
        self.episode_return = 0.0
        self.last_pos_error = 0.0
        self.last_valid_ref = np.array([0.0, 0.0, 1.0], dtype=float)

    def _randomize_dropout_schedule(self) -> None:
        """Randomize dropout start time and duration for domain randomization."""
        if self.dropout_randomize:
            self.dropout_start = np.random.uniform(5.0, 15.0)
            self.dropout_duration = np.random.uniform(1.0, 8.0)
        else:
            self.dropout_start = 5.0
            self.dropout_duration = 4.0

    def _build_state(self) -> np.ndarray:
        """Build 28-dimensional state vector."""
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

        # Dropout flag
        dropout_flag = float(self.env.dropout_mgr.active)

        # Reference
        if self.env.dropout_mgr.active:
            ref_pos = self.last_valid_ref.copy()
            ref_vel = np.zeros(3, dtype=float)
        else:
            ref_pos, ref_vel = self.env.current_reference()
            ref_pos = np.asarray(ref_pos, dtype=float)
            ref_vel = np.asarray(ref_vel, dtype=float)

        state = np.concatenate([
            pos, vel, att, rate,
            std_pos, std_vel, std_att, std_rate,
            [dropout_flag],
            ref_pos, ref_vel
        ]).astype(np.float32)

        return state

    def _action_to_control(self, action: np.ndarray) -> dict:
        """Convert RL action to control commands.

        Action: [thrust_norm, roll_des, pitch_des, yaw_rate_des] (all in [-1, 1])
        Maps to attitude+thrust commands.
        """
        thrust_norm, roll_des, pitch_des, yaw_rate_des = action

        # Map normalized commands to physical ranges
        hover_thrust = self.quad.m * self.quad.g
        thrust_cmd = hover_thrust * (0.9 + 0.4 * float(thrust_norm))  # [0.5mg, 1.3mg]
        thrust_cmd = np.clip(thrust_cmd, 0.5 * hover_thrust, 1.3 * hover_thrust)

        roll_des = float(roll_des) * self.quad.max_angle
        pitch_des = float(pitch_des) * self.quad.max_angle
        yaw_rate_des = float(yaw_rate_des) * self.quad.max_rate

        # Use attitude PID to compute rates + motor commands
        rates_des = self.quad.attitude_pid(roll_des, pitch_des, 0.0)
        rates_des[2] = np.clip(yaw_rate_des, -self.quad.max_rate, self.quad.max_rate)

        tau = self.quad.rate_pid(rates_des)
        omega = self.quad.mixer(thrust_cmd, tau)

        return {
            "thrust_cmd": thrust_cmd,
            "rates_des": rates_des,
            "omega_cmd": omega,
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

        if self.dropout_triggered and t >= self.dropout_start + self.dropout_duration:
            self.env.clear_dropout()
            self.dropout_done = True

        is_drop = bool(self.env.dropout_mgr.active)

        # Reference trajectory
        if is_drop:
            pos_ref = self.last_valid_ref.copy()
            vel_ref = np.zeros(3, dtype=float)
        else:
            pos_ref, vel_ref = self.env.current_reference()
            pos_ref = np.asarray(pos_ref, dtype=float)
            vel_ref = np.asarray(vel_ref, dtype=float)
            self.last_valid_ref = pos_ref.copy()

        # Control: PID during normal flight, RL during dropout
        if is_drop:
            # RL policy controls during dropout
            rl_ctrl = self._action_to_control(action)
            self.quad.inject_external_state(
                self.last_ctrl_pos, np.zeros(3, dtype=float),
                self.ekf.ekf.attitude, self.ekf.ekf.body_rates
            )
            ctrl = rl_ctrl
        else:
            # PID during normal flight
            est = self.ekf.ekf.as_dict()
            self.quad.inject_external_state(est["x"], est["v"], est["ang"], est["rate"])
            ctrl = self.quad.step(pos_ref, vel_ref, z_ref=float(pos_ref[2]))
            self.last_ctrl_pos = np.asarray(est["x"]).copy()

        # Physics step
        action_env = np.zeros(4, dtype=np.float32)
        action_env[0] = self._thrust_to_action(ctrl["thrust_cmd"], self.quad.m)
        action_env[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

        obs, reward_env, done, truncated, info = self.env.step(action_env)

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

        # Compute reward (survival + tracking during dropout)
        pos_error = np.linalg.norm(self.env.drone.xyz - pos_ref)

        if is_drop:
            cov_trace = float(np.trace(self.ekf.ekf.P))

            reward = (
                0.1  # survival bonus for staying alive
                - 10.0 * pos_error                      # minimize position error
                - 0.01 * cov_trace                      # penalize uncertainty
                - 0.001 * np.linalg.norm(ctrl["rates_des"])  # smooth control
            )

            if self.dropout_done:
                reward += 5.0  # bonus for successfully recovering from dropout
        else:
            # Pre-dropout: survival reward helps agent learn not to crash during PID phase
            reward = 0.05

        self.last_pos_error = pos_error if is_drop else 0.0
        self.episode_return += reward

        # Termination conditions
        done = done or truncated
        crashed = self.env.drone.xyz[2] < 0.05  # below ground
        done = done or crashed

        state = self._build_state()
        info["episode_return"] = float(self.episode_return)
        info["crashed"] = crashed
        info["dropout_active"] = is_drop
        info["pos_error"] = float(pos_error)
        info["drone_z"] = float(self.env.drone.xyz[2])
        info["cov_trace"] = float(np.trace(self.ekf.ekf.P))
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

        return state, float(reward), done, False, info

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

                    mission = FlightMission(default_z=1.0, ground_z=0.0)
                    mission.add_takeoff(duration=3.0, target_z=1.0)
                    mission.add_circle(duration=12.0, radius=0.5, period=12.0, z=1.0, center_xy=(0.0, 0.0))
                    mission.add_hover(duration=2.0, z=1.0)
                    mission.add_landing(duration=6.0, ground_z=0.055)

                    self.env = DroneFollowPathDropoutMissionEnv(
                        trajectory_fn=mission,
                        physics="PyBulletPhysics",
                        control_mode="AttitudeRate",
                        drone_model="cf21x_bullet",
                        dropout_mode="NONE",
                        render_mode=None,
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

        self._randomize_dropout_schedule()

        state = self._build_state()
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
