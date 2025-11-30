"""
Follow-path environment for Crazyflie in PyBullet.

Author: Lawrence Wontumi (2025)
"""

import logging
import numpy as np

from phoenix_drone_simulation.envs.base import DroneBaseEnv
from AI_UAV_Tests.trajectories_library import Trajectories as path


class DroneFollowPathEnv(DroneBaseEnv):
    """
    Train the drone to follow an arbitrary trajectory (square, circle, helix, etc.).
    """

    def __init__(self,
    trajectory_fn=None,
    control_mode: str = "PWM",
    log_errors: bool = True,
    **kwargs,):
        if trajectory_fn is None:
            trajectory_fn = path.circle_traj

        self.trajectory_fn = trajectory_fn

        # reward / termination params
        self.done_dist_threshold = 0.3
        self.penalty_action = 1e-4
        self.penalty_velocity = 1e-4
        self.penalty_spin = 1e-4
        self.penalty_terminal = 100.0
        self.ARP = 1e-3

        self.log_errors = log_errors
        self.error_log = []
        self.target_pos = np.zeros(3)

        super().__init__(
            physics="PyBulletPhysics",
            control_mode=control_mode,
            drone_model="cf21x_bullet",
            init_xyz=np.array([0.0, 0.0, 1.0], dtype=float),
            init_rpy=np.zeros(3, dtype=float),
            init_xyz_dot=np.zeros(3, dtype=float),
            init_rpy_dot=np.zeros(3, dtype=float),
            observation_frequency=100,
            sim_freq=200,
            **kwargs,
        )


    # --------------------------------------------------
    # REQUIRED ABSTRACT METHODS
    # --------------------------------------------------

    def _setup_task_specifics(self):
        """Nothing special to spawn for this task."""
        pass

    def get_reference_trajectory(self):
        """
        Return a long horizon reference array.
        DroneBaseEnv only calls this if you use it,
        but we define it to satisfy abstract requirements.
        """
        T = 5000
        traj = np.zeros((T, 3))
        for i in range(T):
            t = i / self.SIM_FREQ
            pos, _ = self.trajectory_fn(t)
            traj[i] = pos
        return traj

    def compute_potential(self):
        """Distance to current reference point."""
        return np.linalg.norm(self.drone.xyz - self.target_pos)

    def compute_info(self):
        """Return diagnostics."""
        return {"tracking_error": float(np.linalg.norm(self.drone.xyz - self.target_pos))}

    # --------------------------------------------------
    # MAIN OBSERVATION + REWARD + DONE LOGIC
    # --------------------------------------------------

    def compute_observation(self):
        t = self.iteration / self.SIM_FREQ
        pos_ref, vel_ref = self.trajectory_fn(t)
        self.target_pos = pos_ref

        e_pos = pos_ref - self.drone.xyz
        e_vel = vel_ref - self.drone.xyz_dot

        if self.log_errors:
            self.error_log.append([t, *e_pos, float(np.linalg.norm(e_pos))])

        obs = np.concatenate(
            [
                self.drone.xyz,
                self.drone.quaternion,
                self.drone.xyz_dot,
                self.drone.rpy_dot,
                e_pos,
                e_vel,
            ]
        )
        return obs

    def compute_reward(self, action):
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)

        penalties = (
            self.penalty_action * np.linalg.norm(action)
            + self.penalty_velocity * np.linalg.norm(self.drone.xyz_dot)
            + self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)
        )

        reward = -dist - penalties
        if self.compute_done():
            reward -= self.penalty_terminal

        return reward

    def compute_done(self):
        return np.linalg.norm(self.drone.xyz - self.target_pos) > self.done_dist_threshold

    def task_specific_reset(self):
        # reset position to (0,0,1)
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=np.array([0.0, 0.0, 1.0]),
            ornObj=self.init_quaternion,
        )

    # --------------------------------------------------
    # Extra helper
    # --------------------------------------------------

    def plot_error(self):
        import matplotlib.pyplot as plt

        arr = np.array(self.error_log)
        if arr.size == 0:
            print("No error data recorded.")
            return None

        t, ex, ey, ez, err_norm = arr.T

        plt.figure(figsize=(8, 5))
        plt.plot(t, ex, label="X error")
        plt.plot(t, ey, label="Y error")
        plt.plot(t, ez, label="Z error")
        plt.plot(t, err_norm, label="‖Error‖", linewidth=2)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return arr
