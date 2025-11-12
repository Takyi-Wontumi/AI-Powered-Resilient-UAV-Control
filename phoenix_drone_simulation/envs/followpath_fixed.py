"""
Fixed, self-contained Follow-Path environment for side-by-side testing.

- Does not modify your original file.
- Implements missing abstract methods from DroneBaseEnv.
- Fixes np.concatenate typo and moves plot_error inside class.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple

from phoenix_drone_simulation.envs.base import DroneBaseEnv


class DroneFollowPathEnv(DroneBaseEnv):
    """Follow an arbitrary reference trajectory (pos, vel) over time.

    trajectory_fn: function t -> (pos_ref[3], vel_ref[3])
    """

    def __init__(
        self,
        trajectory_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
        control_mode: str = "PWM",
        log_errors: bool = True,
        done_dist_threshold: float = 0.3,
        penalty_action: float = 1e-4,
        penalty_velocity: float = 1e-4,
        penalty_spin: float = 1e-4,
        penalty_terminal: float = 100.0,
        ARP: float = 1e-3,
        observation_frequency: int = 100,
        sim_freq: int = 200,
        **kwargs,
    ) -> None:
        # task parameters
        self.trajectory_fn = trajectory_fn
        self.done_dist_threshold = float(done_dist_threshold)
        self.penalty_action = float(penalty_action)
        self.penalty_velocity = float(penalty_velocity)
        self.penalty_spin = float(penalty_spin)
        self.penalty_terminal = float(penalty_terminal)
        self.ARP = float(ARP)

        # logging
        self.log_errors = log_errors
        self.error_log = []  # list of [t, ex, ey, ez, ||e||]

        # placeholders set in _setup_task_specifics
        self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._target_visual_id: Optional[int] = None

        # base init: supply required init state vectors
        init_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        init_rpy = np.zeros(3, dtype=np.float32)
        init_xyz_dot = np.zeros(3, dtype=np.float32)
        init_rpy_dot = np.zeros(3, dtype=np.float32)

        super().__init__(
            physics="PyBulletPhysics",
            control_mode=control_mode,
            drone_model="cf21x_bullet",
            init_xyz=init_xyz,
            init_rpy=init_rpy,
            init_xyz_dot=init_xyz_dot,
            init_rpy_dot=init_rpy_dot,
            observation_frequency=observation_frequency,
            sim_freq=sim_freq,
            **kwargs,
        )

    # ---------------- Base required implementations -----------------
    def _setup_task_specifics(self):
        # Visual marker for the current target point
        self._target_visual_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.bc.createVisualShape(
                self.bc.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0.95, 0.1, 0.05, 0.6],
            ),
            basePosition=self.target_pos,
        )

        # Camera
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-70,
        )

    def compute_info(self) -> dict:
        return {}

    def compute_potential(self) -> float:
        # Shaping potential: negative distance to reference point
        return -float(np.linalg.norm(self.drone.xyz - self.target_pos))

    def get_reference_trajectory(self):
        return self.trajectory_fn

    # ---------------- Task logic -----------------
    def compute_observation(self) -> np.ndarray:
        # time in seconds from simulation ticks
        t = self.iteration / self.SIM_FREQ
        pos_ref, vel_ref = self.trajectory_fn(t)
        pos_ref = np.asarray(pos_ref, dtype=np.float32)
        vel_ref = np.asarray(vel_ref, dtype=np.float32)
        self.target_pos = pos_ref

        # update visual
        if self._target_visual_id is not None:
            self.bc.resetBasePositionAndOrientation(
                self._target_visual_id, pos_ref, self.init_quaternion
            )

        error_to_ref = pos_ref - self.drone.xyz
        error_to_vel = vel_ref - self.drone.xyz_dot  # kept for potential future use

        if self.log_errors:
            self.error_log.append([t, *error_to_ref, float(np.linalg.norm(error_to_ref))])

        obs = np.concatenate(
            [
                self.drone.xyz,
                self.drone.quaternion,
                self.drone.xyz_dot,
                self.drone.rpy_dot,
                error_to_ref,
            ]
        ).astype(np.float32)

        return obs

    def compute_reward(self, action: np.ndarray) -> float:
        dist = float(np.linalg.norm(self.drone.xyz - self.target_pos))
        penalties = (
            self.penalty_action * float(np.linalg.norm(action))
            + self.penalty_velocity * float(np.linalg.norm(self.drone.xyz_dot))
            + self.penalty_spin * float(np.linalg.norm(self.drone.rpy_dot))
        )

        reward = -dist - penalties
        if self.compute_done():
            reward -= self.penalty_terminal
        return float(reward)

    def compute_done(self) -> bool:
        return bool(np.linalg.norm(self.drone.xyz - self.target_pos) > self.done_dist_threshold)

    def task_specific_reset(self):
        # Reset drone pose and clear logs
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id, posObj=np.array([0.0, 0.0, 1.0]), ornObj=self.init_quaternion
        )
        self.error_log = []

        # Reset target visual to initial desired position
        try:
            pos_ref0, _ = self.trajectory_fn(0.0)
            self.target_pos = np.asarray(pos_ref0, dtype=np.float32)
        except Exception:
            self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if self._target_visual_id is not None:
            self.bc.resetBasePositionAndOrientation(
                self._target_visual_id, self.target_pos, self.init_quaternion
            )

    # ---------------- Utilities -----------------
    def plot_error(self):
        import matplotlib.pyplot as plt

        arr = np.array(self.error_log, dtype=float)
        if arr.size == 0:
            print("No error data recorded.")
            return None

        t, ex, ey, ez, err_norm = arr.T
        plt.figure(figsize=(8, 5))
        plt.plot(t, ex, label="X error", color="tab:red")
        plt.plot(t, ey, label="Y error", color="tab:green")
        plt.plot(t, ez, label="Z error", color="tab:blue", linestyle="--", linewidth=1.5)
        plt.plot(t, err_norm, label="||Error||", color="k", linewidth=2)
        plt.title("Trajectory Tracking Error vs Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        return arr

